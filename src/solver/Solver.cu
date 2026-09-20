#include "solver/Solver.h"
#include "core/Check.h"
#include "core/CudaCheck.h"
#include "field/Reduce.h"
#include "solver/SwapGuard.h"

#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// Per-step device synchronisation policy (v2.18.0):
// All solver kernels are issued on the CUDA default stream and are therefore
// ordered without explicit synchronisation — the per-step DeviceSynchronize
// only serialised the host against the GPU and cost a full pipeline drain
// (expensive on WSL2).  It is kept ONLY when the equation was given a
// non-default stream (cross-stream ordering is then the caller's contract).
// Host-side readers (downloads, reductions) block on the stream anyway.
// ---------------------------------------------------------------------------
static void syncIfStreamed(const Equation& eq) {
    if (eq.stream()) CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// GPU kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// Euler update:  dst[i] += coeff * src[i]   (scale-accumulate)
// Used both for Euler (coeff = dt) and inside RK4 stage assembly.
// ---------------------------------------------------------------------------
__global__ void kernel_axpy(Real* dst, const Real* src,
                             double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] += coeff * src[tid];
}

// ---------------------------------------------------------------------------
// Copy:  dst[i] = src[i]
// ---------------------------------------------------------------------------
__global__ void kernel_copy(Real* dst, const Real* src, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

// ---------------------------------------------------------------------------
// RK4 phi_tmp assembly:  phi_tmp = phi + coeff * k_i
// ---------------------------------------------------------------------------
__global__ void kernel_rk4_tmp(Real*       phi_tmp,
                                const Real* phi,
                                const Real* k,
                                double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) phi_tmp[tid] = phi[tid] + coeff * k[tid];
}

// ---------------------------------------------------------------------------
// RK4 final update:  phi += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
// ---------------------------------------------------------------------------
__global__ void kernel_rk4_update(Real*       phi,
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
// Helper: allocate a scratch Field matching unknown's layout
// ===========================================================================
static ScalarField makeScratch(const ScalarField& ref, const std::string& tag) {
    ScalarField f(ref.mesh, tag, ref.ghost);
    f.allocDevice();
    return f;
}

// ===========================================================================
// Constructor
// ===========================================================================

Solver::Solver(Equation&                       equation,
               std::vector<BoundaryCondition*> bcs,
               double                          dt_,
               TimeScheme                      scheme_)
    : dt(dt_)
    , scheme(scheme_)
    , equation_(equation)
    , bcs_(std::move(bcs))
    , rhs_(makeScratch(equation.unknown, equation.unknown.name + "_rhs"))
    , k1_(makeScratch(equation.unknown, "_k1"))
    , k2_(makeScratch(equation.unknown, "_k2"))
    , k3_(makeScratch(equation.unknown, "_k3"))
    , k4_(makeScratch(equation.unknown, "_k4"))
    , phi_tmp_(makeScratch(equation.unknown, "_phi_tmp"))
    , use_rk4_(scheme_ == TimeScheme::RK4)
{
    check::checkPositive(dt_, "dt", "Solver::Solver");
    check::checkFiniteScalar(dt_, "dt", "Solver::Solver");
    // Ensure unknown field is on device
    if (!equation_.unknown.deviceAllocated())
        throw std::runtime_error(
            "Solver: equation.unknown must have device memory allocated "
            "before constructing the Solver.");
    bcBatch_.build(equation_.unknown, bcs_);
}

// ---------------------------------------------------------------------------
// Multi-step constructor
// ---------------------------------------------------------------------------
Solver::Solver(std::vector<SolverStep> steps,
               double                  dt_,
               TimeScheme              scheme_)
    : dt(dt_)
    , scheme(scheme_)
    , equation_(*steps.at(0).equation)          // dummy ref: first equation
    , bcs_({})
    , rhs_(makeScratch(steps[0].equation->unknown, "_ms_rhs"))
    , k1_(makeScratch(steps[0].equation->unknown, "_k1"))
    , k2_(makeScratch(steps[0].equation->unknown, "_k2"))
    , k3_(makeScratch(steps[0].equation->unknown, "_k3"))
    , k4_(makeScratch(steps[0].equation->unknown, "_k4"))
    , phi_tmp_(makeScratch(steps[0].equation->unknown, "_phi_tmp"))
    , use_rk4_(false)
    , multiStep_(true)
    , steps_(std::move(steps))
{
    check::checkPositive(dt_, "dt", "Solver::Solver(multi-step)");
    check::checkFiniteScalar(dt_, "dt", "Solver::Solver(multi-step)");
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        auto& s = steps_[i];
        if (!s.equation->unknown.deviceAllocated())
            throw std::runtime_error(
                "Solver (multi-step): all equation unknowns must have device "
                "memory allocated before constructing the Solver.");
        if (s.type == EquationType::TRANSIENT) {
            stepScratch_.push_back(
                std::make_unique<ScalarField>(
                    makeScratch(s.equation->unknown,
                                s.equation->name + "_rhs")));
        } else {
            stepScratch_.push_back(nullptr);
        }
        stepBatches_.push_back(std::make_unique<BCBatch>());
        stepBatches_.back()->build(*s.sourceField, s.bcs);
    }
}

// ===========================================================================
// Boundary conditions
// ===========================================================================

void Solver::applyBCsGPU() {
    bcBatch_.applyOnGPU(equation_.unknown);
}

void Solver::applyBCsCPU() {
    for (auto* bc : bcs_) bc->applyOnCPU(equation_.unknown);
}

// ===========================================================================
// Euler path
// ===========================================================================

void Solver::eulerUpdateGPU() {
    // unknown.d_curr += dt * rhs.d_curr
    int n = static_cast<int>(equation_.unknown.storedSize);
    kernel_axpy<<<(n + 255) / 256, 256>>>(
        equation_.unknown.d_curr, rhs_.d_curr, dt, n);
    CUDA_CHECK(cudaGetLastError());
}

void Solver::eulerUpdateCPU() {
    auto& phi  = equation_.unknown.curr;
    auto& rhs  = rhs_.curr;
    for (std::size_t i = 0; i < phi.size(); ++i)
        phi[i] += dt * rhs[i];
}

// ===========================================================================
// RK4 GPU path
//
// Classical RK4 for  dphi/dt = f(phi, t):
//   k1 = f(phi)
//   k2 = f(phi + dt/2 * k1)
//   k3 = f(phi + dt/2 * k2)
//   k4 = f(phi +  dt  * k3)
//   phi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//
// Each ki is stored in ki_.d_curr.
// phi_tmp_.d_curr holds the stage-shifted phi for RHS evaluation.
// BCs are applied on phi_tmp before each RHS call.
// ===========================================================================

// Helper: copy phi into phi_tmp then apply BCs on phi_tmp
static void prepTmpGPU(ScalarField& phi_tmp, const ScalarField& phi,
                        const BCBatch& bcs,
                        const Real* k, double coeff)
{
    int n = static_cast<int>(phi.storedSize);
    if (k && coeff != 0.0) {
        kernel_rk4_tmp<<<(n + 255) / 256, 256>>>(
            phi_tmp.d_curr, phi.d_curr, k, coeff, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        kernel_copy<<<(n + 255) / 256, 256>>>(phi_tmp.d_curr, phi.d_curr, n);
        CUDA_CHECK(cudaGetLastError());
    }
    bcs.applyOnGPU(phi_tmp);
}

void Solver::rk4AdvanceGPU() {
    int    n   = static_cast<int>(equation_.unknown.storedSize);
    double dt2 = dt * 0.5;
    double dt6 = dt / 6.0;

    ScalarField& phi = equation_.unknown;
    applyBCsGPU();
    equation_.computeRHS(k1_);   // k1_.d_curr = rhs evaluated at phi

    // k2 = f(phi + dt/2 * k1);
    // Point the unknown's d_curr at phi_tmp_ for the stage evaluation —
    // Equation::computeRHS reads term.field->d_curr.  ScopedDCurrSwap
    // restores the original pointer even if computeRHS throws.
    prepTmpGPU(phi_tmp_, phi, bcBatch_, k1_.d_curr, dt2);
    {
        ScopedDCurrSwap swap(phi, phi_tmp_.d_curr);
        equation_.computeRHS(k2_);
    }

    // k3 = f(phi + dt/2 * k2)
    prepTmpGPU(phi_tmp_, phi, bcBatch_, k2_.d_curr, dt2);
    {
        ScopedDCurrSwap swap(phi, phi_tmp_.d_curr);
        equation_.computeRHS(k3_);
    }

    // k4 = f(phi + dt * k3)
    prepTmpGPU(phi_tmp_, phi, bcBatch_, k3_.d_curr, dt);
    {
        ScopedDCurrSwap swap(phi, phi_tmp_.d_curr);
        equation_.computeRHS(k4_);
    }

    // Final update: phi += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    kernel_rk4_update<<<(n + 255) / 256, 256>>>(
        phi.d_curr,
        k1_.d_curr, k2_.d_curr, k3_.d_curr, k4_.d_curr,
        dt6, n);
    CUDA_CHECK(cudaGetLastError());
    syncIfStreamed(equation_);
}

// ===========================================================================
// RK4 CPU path
// ===========================================================================

void Solver::rk4AdvanceCPU() {
    auto&       phi  = equation_.unknown.curr;
    auto&       tmp  = phi_tmp_.curr;
    auto&       k1c  = k1_.curr;
    auto&       k2c  = k2_.curr;
    auto&       k3c  = k3_.curr;
    auto&       k4c  = k4_.curr;
    std::size_t N    = phi.size();
    double dt2 = dt * 0.5, dt6 = dt / 6.0;

    // k1
    applyBCsCPU();
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k1c.begin());

    // k2
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt2 * k1c[i];
    applyBCsCPU();   // BCs on tmp via swap trick below
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k2c.begin());
    std::swap(phi, tmp);

    // k3
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt2 * k2c[i];
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k3c.begin());
    std::swap(phi, tmp);

    // k4
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt * k3c[i];
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k4c.begin());
    std::swap(phi, tmp);

    // combine
    for (std::size_t i = 0; i < N; ++i)
        phi[i] += dt6 * (k1c[i] + 2.0*k2c[i] + 2.0*k3c[i] + k4c[i]);
}

// ===========================================================================
// Multi-step advance (Euler only)
//
// For each SolverStep in order:
//   1. Apply bcs to sourceField (ghost cell refresh).
//   2. STEADY:    equation.unknown = RHS  (computeRHS writes directly to unknown)
//   3. TRANSIENT: scratch = RHS, then unknown += dt * scratch
// After all steps, call advanceTimeLevelGPU for TRANSIENT unknowns.
// ===========================================================================

void Solver::multiStepAdvanceGPU() {
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        auto& s = steps_[i];
        stepBatches_[i]->applyOnGPU(*s.sourceField);

        if (s.type == EquationType::STEADY) {
            // Write RHS directly into the unknown field (overwrite)
            s.equation->computeRHS(s.equation->unknown);
        } else {
            // Compute RHS into scratch, then axpy
            s.equation->computeRHS(*stepScratch_[i]);
            int n = static_cast<int>(s.equation->unknown.storedSize);
            kernel_axpy<<<(n + 255) / 256, 256>>>(
                s.equation->unknown.d_curr,
                stepScratch_[i]->d_curr,
                dt, n);
            CUDA_CHECK(cudaGetLastError());
            syncIfStreamed(*s.equation);
        }
    }
    for (auto& s : steps_)
        if (s.type == EquationType::TRANSIENT)
            s.equation->unknown.advanceTimeLevelGPU();
}

void Solver::multiStepAdvanceCPU() {
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        auto& s = steps_[i];
        for (auto* bc : s.bcs) bc->applyOnCPU(*s.sourceField);

        if (s.type == EquationType::STEADY) {
            s.equation->computeRHSCPU(s.equation->unknown);
        } else {
            s.equation->computeRHSCPU(*stepScratch_[i]);
            auto& phi = s.equation->unknown.curr;
            auto& rhs = stepScratch_[i]->curr;
            for (std::size_t j = 0; j < phi.size(); ++j)
                phi[j] += dt * rhs[j];
        }
    }
    for (auto& s : steps_)
        if (s.type == EquationType::TRANSIENT)
            s.equation->unknown.advanceTimeLevelCPU();
}

// ===========================================================================
// Public advance / advanceCPU
// ===========================================================================

void Solver::enableAdaptiveDt(const AdaptiveDt& opts) {
    if (use_rk4_ || multiStep_)
        throw std::invalid_argument(
            "Solver::enableAdaptiveDt: only single-equation Euler mode is "
            "supported (RK4 needs an embedded error estimate; multi-step "
            "updates fields before all RHS are known).");
    opts.validate();
    adapt_ = opts;
    adapt_.enabled = true;
}

// Host-side max|value| over physical cells (CPU fallback path)
static double maxAbsPhysicalCPU(const ScalarField& f) {
    double m = 0.0;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i)
        m = std::max(m, static_cast<double>(std::fabs(
                f.curr[static_cast<std::size_t>(f.index(i, j, k))])));
    return m;
}

void Solver::advance() {
    check::checkPositive(dt, "dt", "Solver::advance");   // dt is public/mutable
    if (multiStep_) {
        multiStepAdvanceGPU();   // handles advanceTimeLevel internally
    } else {
        if (use_rk4_) {
            rk4AdvanceGPU();
        } else {
            applyBCsGPU();
            equation_.computeRHS(rhs_);
            if (adapt_.enabled) {
                adapt_.lastMaxRate = reduce::fieldMaxAbs(rhs_);
                dt = adapt_.propose(dt, adapt_.lastMaxRate);
            }
            eulerUpdateGPU();
            syncIfStreamed(equation_);
        }
        equation_.unknown.advanceTimeLevelGPU();
    }
    ++step;
    time += dt;

    if (adapt_.enabled && adapt_.nanCheckEvery > 0
        && step % adapt_.nanCheckEvery == 0
        && reduce::fieldHasNonFinite(equation_.unknown))
        throw std::runtime_error(
            "Solver::advance: NaN/Inf detected in unknown '"
            + equation_.unknown.name + "' at step "
            + std::to_string(step) + " (t=" + std::to_string(time) + ")");

    // Default-on health sentinel (HealthCheck.h) — independent of adaptive dt.
    if (health.due(step)) {
        if (multiStep_) {
            for (auto& s : steps_)
                check::checkFieldHealth(s.equation->unknown, step, time,
                                        health.maxAbsLimit);
        } else {
            check::checkFieldHealth(equation_.unknown, step, time,
                                    health.maxAbsLimit);
        }
    }
}

void Solver::advanceCPU() {
    check::checkPositive(dt, "dt", "Solver::advanceCPU");
    if (multiStep_) {
        multiStepAdvanceCPU();   // handles advanceTimeLevel internally
    } else {
        if (use_rk4_) {
            rk4AdvanceCPU();
        } else {
            applyBCsCPU();
            equation_.computeRHSCPU(rhs_);
            if (adapt_.enabled) {
                adapt_.lastMaxRate = maxAbsPhysicalCPU(rhs_);
                dt = adapt_.propose(dt, adapt_.lastMaxRate);
            }
            eulerUpdateCPU();
        }
        equation_.unknown.advanceTimeLevelCPU();
    }
    ++step;
    time += dt;

    // CPU-path sentinel: scan the host copy (authoritative here).
    if (health.due(step)) {
        if (multiStep_) {
            for (auto& s : steps_)
                check::checkFieldHealthCPU(s.equation->unknown, step, time,
                                           health.maxAbsLimit);
        } else {
            check::checkFieldHealthCPU(equation_.unknown, step, time,
                                       health.maxAbsLimit);
        }
    }
}

// ===========================================================================
// run
// ===========================================================================

void Solver::run(int nSteps,
                 int callbackEvery,
                 std::function<void(const Solver&)> callback)
{
    for (int s = 0; s < nSteps; ++s) {
        advance();
        if (callback && callbackEvery > 0 && (step % callbackEvery == 0))
            callback(*this);
    }
}

} // namespace PhiX
