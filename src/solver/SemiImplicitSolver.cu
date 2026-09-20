#include "solver/SemiImplicitSolver.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// b = x + dt·r
__global__ void kernel_form_b(Real* b, const Real* x, const Real* r,
                              Real dt, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = x[i] + dt * r[i];
}

ScalarField makeScratch(const ScalarField& ref, const char* name) {
    ScalarField f(ref.mesh, name, ref.ghost);
    f.allocDevice();
    CUDA_CHECK(cudaMemset(f.d_curr, 0, f.storedSize * sizeof(Real)));
    return f;
}

} // namespace

SemiImplicitSolver::SemiImplicitSolver(Equation& eqExplicit,
                                       std::vector<BoundaryCondition*> bcs,
                                       LinearOperator& L,
                                       double dt_,
                                       CGOptions cg)
    : dt(dt_)
    , eq_(eqExplicit)
    , bcs_(std::move(bcs))
    , L_(L)
    , cgOpts_(cg)
    , rhs_(makeScratch(eqExplicit.unknown, "_si_rhs"))
    , b_(makeScratch(eqExplicit.unknown, "_si_b"))
    , cg_(eqExplicit.unknown.mesh, eqExplicit.unknown.ghost)
{
    if (!eq_.unknown.deviceAllocated())
        throw std::runtime_error(
            "SemiImplicitSolver: equation.unknown must have device memory "
            "allocated before constructing the solver.");
    if (eq_.unknown.ghost < L_.ghostRequired())
        throw std::invalid_argument(
            "SemiImplicitSolver: operator needs ghost >= "
            + std::to_string(L_.ghostRequired()));
    bcBatch_.build(eq_.unknown, bcs_);
}

void SemiImplicitSolver::advance()
{
    ScalarField& phi = eq_.unknown;
    const int n = static_cast<int>(phi.storedSize);

    // 1. ghosts for the explicit stencils
    bcBatch_.applyOnGPU(phi);

    // 2. b = φⁿ + dt·N(φⁿ)   (N ≡ 0 when no explicit RHS was set)
    if (eq_.hasRHS()) {
        eq_.computeRHS(rhs_);
        kernel_form_b<<<(n + 255) / 256, 256>>>(
            b_.d_curr, phi.d_curr, rhs_.d_curr, static_cast<Real>(dt), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(b_.d_curr, phi.d_curr, n * sizeof(Real),
                              cudaMemcpyDeviceToDevice));
    }

    // 3. (I − dt·L) φⁿ⁺¹ = b, initial guess φⁿ (solved in place).
    // No trailing device sync: the CG convergence checkpoint already
    // drained the pipeline, and host readers block on the stream anyway.
    last_ = cg_.solve(L_, dt, phi, b_, cgOpts_.relTol, cgOpts_.maxIter);

    phi.advanceTimeLevelGPU();

    ++step;
    time += dt;
}

void SemiImplicitSolver::run(int nSteps,
                             int callbackEvery,
                             std::function<void(const SemiImplicitSolver&)> callback)
{
    for (int s = 0; s < nSteps; ++s) {
        advance();
        if (callback && callbackEvery > 0 && (step % callbackEvery == 0))
            callback(*this);
    }
}

} // namespace PhiX
