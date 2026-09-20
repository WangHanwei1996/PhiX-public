// ---------------------------------------------------------------------------
// module_precond — preconditioner layer for the matrix-free CG
// (solver/Preconditioner.h + the PCG path of ConjugateGradient).
//
// Consistent-system methodology (as module_linsolve): b = A·x_ref with the
// same matrix-free operator, solve from zero guess, require x → x_ref.
//
//   1. DiagonalPreconditioner (constant diag) — for a constant-coefficient
//      operator M is a scalar rescaling: PCG must reproduce the plain-CG
//      solution and iteration count (±checkEvery).
//   2. ChebyshevPreconditioner on a stiff Helmholtz (σD/dx² = 500, 2D
//      periodic): iterations must drop below half of plain CG at equal
//      tolerance, solution must match x_ref.
//   3. σ-change through a REPLAYED burst graph: same x/b/L/M buffers, new σ
//      (and new consistent rhs) — the second solve must still converge to
//      its own x_ref, proving the device-slot σ propagation inside the
//      captured graph (including the Chebyshev interval recomputation).
//   4. No-flux BCs with Chebyshev — BC-agnostic convergence.
//   5. DiagonalPreconditioner (field variant) == const variant.
// ---------------------------------------------------------------------------

#include "solver/LinearSolver.h"
#include "solver/Preconditioner.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static int nPass = 0;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
    ++nPass;
}

// b = x_ref − σ·L(x_ref)   (device apply, host combine, upload)
static void buildRhs(LinearOperator& L, double sigma,
                     ScalarField& xRef, ScalarField& b) {
    ScalarField Lx(xRef.mesh, "_Lx", xRef.ghost);
    Lx.allocDevice();
    L.apply(xRef, Lx);
    Lx.downloadCurrFromDevice();
    xRef.downloadCurrFromDevice();
    for (std::size_t i = 0; i < b.storedSize; ++i)
        b.curr[i] = xRef.curr[i] - static_cast<Real>(sigma) * Lx.curr[i];
    if (!b.deviceAllocated()) b.allocDevice();
    b.uploadAllToDevice();
}

static double maxErrVsRef(ScalarField& x, const ScalarField& xRef) {
    x.downloadCurrFromDevice();
    double err = 0.0;
    for (int k = 0; k < x.mesh.n[2]; ++k)
    for (int j = 0; j < x.mesh.n[1]; ++j)
    for (int i = 0; i < x.mesh.n[0]; ++i) {
        const std::size_t idx = static_cast<std::size_t>(x.index(i, j, k));
        err = std::max(err, std::fabs(
            static_cast<double>(x.curr[idx]) - xRef.curr[idx]));
    }
    return err;
}

static ScalarField makeDev(const Mesh& m, const char* name, int ghost) {
    ScalarField f(m, name, ghost);
    f.fill(0.0);
    f.allocDevice();
    f.uploadAllToDevice();
    return f;
}

int main() {
try {
    // === 1+2+3: 2D periodic Helmholtz ====================================
    const int    N  = 128;
    const double dx = 1.0 / N;
    const double D  = 1.0;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0,
                                    N, dx, 0.0);
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bcx, &bcy};

    LaplacianOp L(D, bcs);
    const double sigma = 500.0 * dx * dx / D;      // σD/dx² = 500 (stiff)

    // Broadband (full-spectrum) reference: a smooth few-mode x_ref lets the
    // Krylov space capture the solution in O(#modes) iterations regardless of
    // conditioning — a deterministic per-cell hash excites every eigenmode
    // and makes the σD/dx² stiffness actually bite.
    ScalarField xRef = makeDev(mesh, "xref", 1);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            double v = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1))
                     * 43758.5453;
            v -= std::floor(v);
            xRef.curr[xRef.index(i, j, 0)] = static_cast<Real>(v - 0.5);
        }
    xRef.uploadAllToDevice();

    ScalarField b = makeDev(mesh, "b", 1);
    buildRhs(L, sigma, xRef, b);

    ScalarField x = makeDev(mesh, "x", 1);
    ConjugateGradient cg(mesh, 1);

    // --- plain CG baseline ---
    auto r0 = cg.solve(L, sigma, x, b, 1e-10, 2000);
    require(r0.converged, "baseline CG did not converge");
    require(maxErrVsRef(x, xRef) < 1e-7, "baseline CG wrong solution");
    std::printf("  [1] plain CG: %d iters\n", r0.iterations);

    // --- diagonal (constant) PCG: scalar rescale == same Krylov path ---
    DiagonalPreconditioner jac(-2.0 * D * (1.0 / (dx * dx) + 1.0 / (dx * dx)));
    CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
    auto r1 = cg.solve(L, sigma, x, b, 1e-10, 2000, true, &jac);
    require(r1.converged, "diag-PCG did not converge");
    require(maxErrVsRef(x, xRef) < 1e-7, "diag-PCG wrong solution");
    require(std::abs(r1.iterations - r0.iterations) <= 2 * cg.checkEvery,
            "diag-const PCG iteration count deviates from CG");
    std::printf("  [2] diag-const PCG: %d iters (CG %d)\n",
                r1.iterations, r0.iterations);

    // --- Chebyshev PCG: must at least halve the iterations ---
    ChebyshevPreconditioner cheb(
        L, ChebyshevPreconditioner::lambdaMaxCD2(mesh, D), 6, mesh, 1);
    CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
    auto r2 = cg.solve(L, sigma, x, b, 1e-10, 2000, true, &cheb);
    require(r2.converged, "cheb-PCG did not converge");
    require(maxErrVsRef(x, xRef) < 1e-7, "cheb-PCG wrong solution");
    require(r2.iterations * 2 < r0.iterations,
            "cheb-PCG did not halve the CG iteration count ("
            + std::to_string(r2.iterations) + " vs "
            + std::to_string(r0.iterations) + ")");
    std::printf("  [3] cheb(6) PCG: %d iters (CG %d)\n",
                r2.iterations, r0.iterations);

    // --- σ change through the REPLAYED graph (same x/b/L/M buffers) ---
    const double sigma2 = sigma / 8.0;
    buildRhs(L, sigma2, xRef, b);                  // same b buffer
    CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
    auto r3 = cg.solve(L, sigma2, x, b, 1e-10, 2000, true, &cheb);
    require(r3.converged, "cheb-PCG (sigma changed) did not converge");
    require(maxErrVsRef(x, xRef) < 1e-7,
            "cheb-PCG wrong solution after sigma change through graph");
    std::printf("  [4] cheb PCG after sigma/8 via replayed graph: %d iters\n",
                r3.iterations);

    // === 4: no-flux BCs ===================================================
    NoFluxBC nbxL(mesh.facePatch(Axis::X, Side::LOW));
    NoFluxBC nbxH(mesh.facePatch(Axis::X, Side::HIGH));
    NoFluxBC nbyL(mesh.facePatch(Axis::Y, Side::LOW));
    NoFluxBC nbyH(mesh.facePatch(Axis::Y, Side::HIGH));
    std::vector<BoundaryCondition*> nbcs = {&nbxL, &nbxH, &nbyL, &nbyH};
    LaplacianOp Ln(D, nbcs);
    ScalarField xRefN = makeDev(mesh, "xrefn", 1);
    xRefN.initialize([](double X, double Y, double) {
        return std::cos(2.0 * M_PI * X) * std::cos(3.0 * M_PI * Y);
    });
    xRefN.uploadAllToDevice();
    ScalarField bN = makeDev(mesh, "bn", 1);
    buildRhs(Ln, sigma, xRefN, bN);
    ChebyshevPreconditioner chebN(
        Ln, ChebyshevPreconditioner::lambdaMaxCD2(mesh, D), 6, mesh, 1);
    ScalarField xN = makeDev(mesh, "xn", 1);
    ConjugateGradient cgN(mesh, 1);
    auto r4 = cgN.solve(Ln, sigma, xN, bN, 1e-10, 2000, true, &chebN);
    require(r4.converged, "no-flux cheb-PCG did not converge");
    require(maxErrVsRef(xN, xRefN) < 1e-7, "no-flux cheb-PCG wrong solution");
    std::printf("  [5] no-flux cheb PCG: %d iters\n", r4.iterations);

    // === 5: diagonal field variant == const variant =======================
    ScalarField diagF = makeDev(mesh, "diagL", 1);
    diagF.fill(-2.0 * D * (1.0 / (dx * dx) + 1.0 / (dx * dx)));
    diagF.uploadAllToDevice();
    DiagonalPreconditioner jacF(&diagF);
    buildRhs(L, sigma, xRef, b);
    CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
    auto r5 = cg.solve(L, sigma, x, b, 1e-10, 2000, true, &jacF);
    require(r5.converged, "diag-field PCG did not converge");
    require(maxErrVsRef(x, xRef) < 1e-7, "diag-field PCG wrong solution");
    std::printf("  [6] diag-field PCG: %d iters\n", r5.iterations);

    std::printf("module_precond: %d assertions passed\n", nPass);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "module_precond FAILED: %s\n", e.what());
    return 1;
}
}
