#include "solver/VectorSolver.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace PhiX {

// ===========================================================================
// GPU kernels (shared with scalar Solver — redeclared here to keep
// VectorSolver self-contained without a shared internal header)
// ===========================================================================

__global__ void vsolv_axpy(Real* dst, const Real* src,
                            double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] += coeff * src[tid];
}

__global__ void vsolv_copy(Real* dst, const Real* src, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

__global__ void vsolv_rk4_tmp(Real*       phi_tmp,
                               const Real* phi,
                               const Real* k,
                               double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) phi_tmp[tid] = phi[tid] + coeff * k[tid];
}

__global__ void vsolv_rk4_update(Real*       phi,
                                  const Real* k1,
                                  const Real* k2,
                                  const Real* k3,
                                  const Real* k4,
                                  double dt6, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        phi[tid] += dt6 * (k1[tid] + 2.0 * k2[tid] + 2.0 * k3[tid] + k4[tid]);
}

// ===========================================================================
// Helper: build a scratch VectorField matching the unknown's layout
// ===========================================================================
static VectorField makeVectorScratch(const VectorField& ref,
                                     const std::string& tag)
{
    VectorField vf(ref.mesh, tag, ref.nComponents(), ref.ghost);
    for (int c = 0; c < vf.nComponents(); ++c)
        vf[c].allocDevice();
    return vf;
}

// ===========================================================================
// Constructor
// ===========================================================================

VectorSolver::VectorSolver(VectorEquation&                     equation,
                            std::vector<BoundaryCondition*>     bcs,
                            double                              dt_,
                            TimeScheme                          scheme_)
    : dt(dt_)
    , scheme(scheme_)
    , equation_(equation)
    , bcs_(std::move(bcs))
    , use_rk4_(scheme_ == TimeScheme::RK4)
    , rhs_     (makeVectorScratch(equation.unknown, equation.unknown.name + "_rhs"))
    , k1_      (makeVectorScratch(equation.unknown, equation.unknown.name + "_k1" ))
    , k2_      (makeVectorScratch(equation.unknown, equation.unknown.name + "_k2" ))
    , k3_      (makeVectorScratch(equation.unknown, equation.unknown.name + "_k3" ))
    , k4_      (makeVectorScratch(equation.unknown, equation.unknown.name + "_k4" ))
    , phi_tmp_ (makeVectorScratch(equation.unknown, equation.unknown.name + "_tmp"))
{
    const int N = equation.unknown.nComponents();
    for (int c = 0; c < N; ++c) {
        if (!equation_.unknown[c].deviceAllocated())
            throw std::runtime_error(
                "VectorSolver: all components of equation.unknown must have "
                "device memory allocated before constructing VectorSolver.");
    }
}

// ===========================================================================
// Boundary conditions
// ===========================================================================

void VectorSolver::applyBCsGPU() {
    for (auto* bc : bcs_) bc->applyOnGPU(equation_.unknown);
}

void VectorSolver::applyBCsCPU() {
    for (auto* bc : bcs_) bc->applyOnCPU(equation_.unknown);
}

// ===========================================================================
// Euler path
// ===========================================================================

void VectorSolver::eulerUpdateGPU() {
    const int N = equation_.unknown.nComponents();
    for (int c = 0; c < N; ++c) {
        int n = static_cast<int>(equation_.unknown[c].storedSize);
        vsolv_axpy<<<(n + 255) / 256, 256>>>(
            equation_.unknown[c].d_curr, rhs_[c].d_curr, dt, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

void VectorSolver::eulerUpdateCPU() {
    const int N = equation_.unknown.nComponents();
    for (int c = 0; c < N; ++c) {
        auto& phi = equation_.unknown[c].curr;
        auto& rhs = rhs_[c].curr;
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] += dt * rhs[i];
    }
}

// ===========================================================================
// RK4 GPU path
//
// Key requirement: all component d_curr pointers are swapped simultaneously
// before each equation_.computeRHS() call, so that cross-component coupling
// in the RHS is evaluated at a consistent stage state.
// ===========================================================================

// Helper: for all components, build phi_tmp[c] = phi[c] + coeff * k[c],
//         apply BCs on phi_tmp, then swap d_curr pointers of unknown & phi_tmp.
// Returns the saved original pointers so the caller can restore them.
static std::vector<Real*>
prepAndSwapAll(VectorField& phi, VectorField& phi_tmp,
               VectorField* k, double coeff,
               std::vector<BoundaryCondition*>& bcs)
{
    const int N = phi.nComponents();
    std::vector<Real*> orig(N);

    // Build phi_tmp and optionally apply BCs
    for (int c = 0; c < N; ++c) {
        int n = static_cast<int>(phi[c].storedSize);
        if (k != nullptr && coeff != 0.0) {
            vsolv_rk4_tmp<<<(n + 255) / 256, 256>>>(
                phi_tmp[c].d_curr, phi[c].d_curr, (*k)[c].d_curr, coeff, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            vsolv_copy<<<(n + 255) / 256, 256>>>(
                phi_tmp[c].d_curr, phi[c].d_curr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    // Apply BCs to phi_tmp (need to swap first so BCs see phi_tmp through unknown)
    // We do the swap, apply BCs via the unknown alias, then the caller will
    // computeRHS, then restore.
    for (int c = 0; c < N; ++c) {
        orig[c]           = phi[c].d_curr;
        phi[c].d_curr     = phi_tmp[c].d_curr;
    }
    for (auto* bc : bcs) bc->applyOnGPU(phi);   // phi now aliases phi_tmp data
    return orig;
}

static void restoreAll(VectorField& phi, const std::vector<Real*>& orig) {
    for (int c = 0; c < phi.nComponents(); ++c)
        phi[c].d_curr = orig[c];
}

void VectorSolver::rk4AdvanceGPU() {
    const int    N   = equation_.unknown.nComponents();
    double       dt2 = dt * 0.5;
    double       dt6 = dt / 6.0;

    VectorField& phi = equation_.unknown;

    // k1 = f(phi)
    applyBCsGPU();
    equation_.computeRHS(k1_);

    // k2 = f(phi + dt/2 * k1)
    {
        auto orig = prepAndSwapAll(phi, phi_tmp_, &k1_, dt2, bcs_);
        equation_.computeRHS(k2_);
        restoreAll(phi, orig);
    }

    // k3 = f(phi + dt/2 * k2)
    {
        auto orig = prepAndSwapAll(phi, phi_tmp_, &k2_, dt2, bcs_);
        equation_.computeRHS(k3_);
        restoreAll(phi, orig);
    }

    // k4 = f(phi + dt * k3)
    {
        auto orig = prepAndSwapAll(phi, phi_tmp_, &k3_, dt, bcs_);
        equation_.computeRHS(k4_);
        restoreAll(phi, orig);
    }

    // Final update: phi[c] += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    for (int c = 0; c < N; ++c) {
        int n = static_cast<int>(phi[c].storedSize);
        vsolv_rk4_update<<<(n + 255) / 256, 256>>>(
            phi[c].d_curr,
            k1_[c].d_curr, k2_[c].d_curr, k3_[c].d_curr, k4_[c].d_curr,
            dt6, n);
        CUDA_CHECK(cudaGetLastError());
    }
    if (equation_.stream()) CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// RK4 CPU path
// ===========================================================================

void VectorSolver::rk4AdvanceCPU() {
    const int N  = equation_.unknown.nComponents();
    double dt2 = dt * 0.5, dt6 = dt / 6.0;

    // Evaluate RHS by temporarily swapping CPU buffers of all components.
    auto evalRHSIntoK = [&](VectorField& stage_offset, double coeff,
                             VectorField& ki) {
        // Build phi_tmp[c] = phi[c] + coeff * stage_offset[c]
        for (int c = 0; c < N; ++c) {
            auto& phi = equation_.unknown[c].curr;
            auto& ko  = stage_offset[c].curr;
            auto& tmp = phi_tmp_[c].curr;
            for (std::size_t i = 0; i < phi.size(); ++i)
                tmp[i] = phi[i] + coeff * ko[i];
        }
        // Swap all components, apply BCs, compute, swap back
        for (int c = 0; c < N; ++c)
            std::swap(equation_.unknown[c].curr, phi_tmp_[c].curr);
        applyBCsCPU();
        equation_.computeRHSCPU(rhs_);
        for (int c = 0; c < N; ++c) {
            std::copy(rhs_[c].curr.begin(), rhs_[c].curr.end(),
                      ki[c].curr.begin());
        }
        for (int c = 0; c < N; ++c)
            std::swap(equation_.unknown[c].curr, phi_tmp_[c].curr);
    };

    // k1
    applyBCsCPU();
    equation_.computeRHSCPU(rhs_);
    for (int c = 0; c < N; ++c)
        std::copy(rhs_[c].curr.begin(), rhs_[c].curr.end(),
                  k1_[c].curr.begin());

    // k2, k3, k4
    evalRHSIntoK(k1_, dt2, k2_);
    evalRHSIntoK(k2_, dt2, k3_);
    evalRHSIntoK(k3_, dt,  k4_);

    // Combine
    for (int c = 0; c < N; ++c) {
        auto& phi = equation_.unknown[c].curr;
        auto& k1c = k1_[c].curr;
        auto& k2c = k2_[c].curr;
        auto& k3c = k3_[c].curr;
        auto& k4c = k4_[c].curr;
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] += dt6 * (k1c[i] + 2.0*k2c[i] + 2.0*k3c[i] + k4c[i]);
    }
}

// ===========================================================================
// Public advance / advanceCPU
// ===========================================================================

void VectorSolver::advance() {
    if (use_rk4_) {
        rk4AdvanceGPU();
    } else {
        applyBCsGPU();
        equation_.computeRHS(rhs_);
        eulerUpdateGPU();
        if (equation_.stream()) CUDA_CHECK(cudaDeviceSynchronize());
    }
    const int N = equation_.unknown.nComponents();
    for (int c = 0; c < N; ++c)
        equation_.unknown[c].advanceTimeLevelGPU();
    ++step;
    time += dt;
}

void VectorSolver::advanceCPU() {
    if (use_rk4_) {
        rk4AdvanceCPU();
    } else {
        applyBCsCPU();
        equation_.computeRHSCPU(rhs_);
        eulerUpdateCPU();
    }
    const int N = equation_.unknown.nComponents();
    for (int c = 0; c < N; ++c)
        equation_.unknown[c].advanceTimeLevelCPU();
    ++step;
    time += dt;
}

// ===========================================================================
// run
// ===========================================================================

void VectorSolver::run(int nSteps,
                        int callbackEvery,
                        std::function<void(const VectorSolver&)> callback)
{
    for (int s = 0; s < nSteps; ++s) {
        advance();
        if (callback && callbackEvery > 0 && (step % callbackEvery == 0))
            callback(*this);
    }
}

} // namespace PhiX
