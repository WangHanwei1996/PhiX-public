#include "equation/EquationSystem.h"
#include "core/Check.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "solver/SwapGuard.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

// ===========================================================================
// GPU kernels (reuse the same logic as Solver.cu; defined locally here to
// avoid duplicate-symbol issues since both files are compiled separately)
// ===========================================================================

// dst[i] += coeff * src[i]
__global__ static void eqsys_axpy(Real* dst, const Real* src,
                                   double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] += coeff * src[tid];
}

// phi_tmp[i] = phi[i] + coeff * k[i]
__global__ static void eqsys_rk4_tmp(Real*       phi_tmp,
                                      const Real* phi,
                                      const Real* k,
                                      double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) phi_tmp[tid] = phi[tid] + coeff * k[tid];
}

// phi[i] += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)[i]
__global__ static void eqsys_rk4_update(Real*       phi,
                                         const Real* k1,
                                         const Real* k2,
                                         const Real* k3,
                                         const Real* k4,
                                         double dt6, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        phi[tid] += dt6 * (k1[tid] + 2.0*k2[tid] + 2.0*k3[tid] + k4[tid]);
}

// ===========================================================================
// Helpers
// ===========================================================================

static std::unique_ptr<ScalarField> makeScratchPtr(const ScalarField& ref,
                                                    const std::string& tag)
{
    auto p = std::make_unique<ScalarField>(ref.mesh, tag, ref.ghost);
    p->allocDevice();
    return p;
}

// ===========================================================================
// Constructor / add
// ===========================================================================

EquationSystem::EquationSystem(double dt_, TimeScheme scheme_)
    : dt(dt_), scheme(scheme_)
{
    check::checkPositive(dt_, "dt", "EquationSystem::EquationSystem");
    check::checkFiniteScalar(dt_, "dt", "EquationSystem::EquationSystem");
}

void EquationSystem::add(Equation&                       equation,
                         std::vector<BoundaryCondition*> bcs,
                         ScalarField*                    sourceField)
{
    if (initialized_)
        throw std::logic_error(
            "EquationSystem::add() must not be called after the first advance().");

    entries_.push_back({&equation,
                        std::move(bcs),
                        sourceField ? sourceField : &equation.unknown});
}

// ===========================================================================
// Lazy scratch allocation
// ===========================================================================

void EquationSystem::lazyInit_()
{
    if (initialized_) return;
    for (auto& e : entries_) {
        e.batch = std::make_unique<BCBatch>();
        e.batch->build(*e.sourceField, e.bcs);
    }

    initialized_ = true;

    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const auto& e    = entries_[i];
        const std::string tag = e.equation->name;

        if (!e.equation->unknown.deviceAllocated())
            throw std::runtime_error(
                "EquationSystem: equation '" + tag + "' unknown must have "
                "device memory allocated before advance().");

        rhs_.push_back(makeScratchPtr(e.equation->unknown, tag + "_rhs"));

        if (scheme == TimeScheme::RK4) {
            k1_.push_back(makeScratchPtr(e.equation->unknown, tag + "_k1"));
            k2_.push_back(makeScratchPtr(e.equation->unknown, tag + "_k2"));
            k3_.push_back(makeScratchPtr(e.equation->unknown, tag + "_k3"));
            k4_.push_back(makeScratchPtr(e.equation->unknown, tag + "_k4"));
            phi_tmp_.push_back(makeScratchPtr(e.equation->unknown, tag + "_tmp"));
        }
    }
}

// ---------------------------------------------------------------------------
// Per-step device synchronisation policy (v2.18.0): kernels run on the
// default stream and are ordered without explicit sync; a full
// DeviceSynchronize is only needed when some equation uses its own stream.
// ---------------------------------------------------------------------------

void EquationSystem::maybeSync_() const
{
    for (const auto& e : entries_)
        if (e.equation->stream()) {
            CUDA_CHECK(cudaDeviceSynchronize());
            return;
        }
}

// ===========================================================================
// Apply all BCs
// ===========================================================================

void EquationSystem::applyAllBCsGPU_()
{
    for (auto& e : entries_)
        e.batch->applyOnGPU(*e.sourceField);
}

void EquationSystem::applyAllBCsCPU_()
{
    for (auto& e : entries_)
        for (auto* bc : e.bcs)
            bc->applyOnCPU(*e.sourceField);
}

// ===========================================================================
// Euler advance (GPU)
//
// Step 1: apply ALL BCs        (all fields at current time level)
// Step 2: compute ALL RHS      (from current d_curr — no field has been updated)
// Step 3: update ALL fields    (phi += dt * rhs)
// ===========================================================================

void EquationSystem::eulerAdvanceGPU_()
{
    // 1. BCs
    applyAllBCsGPU_();

    // 2. RHS (simultaneously — no field updated yet)
    for (std::size_t i = 0; i < entries_.size(); ++i)
        entries_[i].equation->computeRHS(*rhs_[i]);

    // 2b. Adaptive dt: the stiffest equation controls the step
    if (adapt_.enabled) {
        double rate = 0.0;
        for (auto& r : rhs_)
            rate = std::max(rate, reduce::fieldMaxAbs(*r));
        adapt_.lastMaxRate = rate;
        dt = adapt_.propose(dt, rate);
    }

    // 3. Euler update
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        int n = static_cast<int>(entries_[i].equation->unknown.storedSize);
        eqsys_axpy<<<(n + 255) / 256, 256>>>(
            entries_[i].equation->unknown.d_curr,
            rhs_[i]->d_curr,
            dt, n);
        CUDA_CHECK(cudaGetLastError());
    }
    maybeSync_();

    for (auto& e : entries_)
        e.equation->unknown.advanceTimeLevelGPU();
}

// ===========================================================================
// Euler advance (CPU)
// ===========================================================================

void EquationSystem::eulerAdvanceCPU_()
{
    applyAllBCsCPU_();

    for (std::size_t i = 0; i < entries_.size(); ++i)
        entries_[i].equation->computeRHSCPU(*rhs_[i]);

    if (adapt_.enabled) {
        double rate = 0.0;
        for (auto& r : rhs_) {
            const ScalarField& f = *r;
            for (int k = 0; k < f.mesh.n[2]; ++k)
            for (int j = 0; j < f.mesh.n[1]; ++j)
            for (int i = 0; i < f.mesh.n[0]; ++i)
                rate = std::max(rate, static_cast<double>(std::fabs(
                    f.curr[static_cast<std::size_t>(f.index(i, j, k))])));
        }
        adapt_.lastMaxRate = rate;
        dt = adapt_.propose(dt, rate);
    }

    for (std::size_t i = 0; i < entries_.size(); ++i) {
        auto& phi = entries_[i].equation->unknown.curr;
        auto& rhs = rhs_[i]->curr;
        for (std::size_t j = 0; j < phi.size(); ++j)
            phi[j] += dt * rhs[j];
    }

    for (auto& e : entries_)
        e.equation->unknown.advanceTimeLevelCPU();
}

// ===========================================================================
// RK4 advance (GPU)
//
// Classical RK4 applied to the full coupled system.
// At each stage, ALL fields are shifted simultaneously before ANY RHS is
// evaluated, preserving the time-level consistency of the simultaneous scheme.
//
// Stage naming follows the single-equation pattern in Solver.cu, extended to
// operate on a vector of fields.  The d_curr swap trick (temporarily pointing
// equation.unknown.d_curr at phi_tmp.d_curr) is used so that the existing
// Equation::computeRHS logic reads the stage-shifted values transparently.
// ===========================================================================

void EquationSystem::rk4AdvanceGPU_()
{
    const std::size_t N  = entries_.size();
    const double dt2     = dt * 0.5;
    const double dt6     = dt / 6.0;

    // --- k1: evaluate RHS at current phi ---
    applyAllBCsGPU_();
    for (std::size_t i = 0; i < N; ++i)
        entries_[i].equation->computeRHS(*k1_[i]);

    // --- k2: phi_tmp[i] = phi[i] + dt/2 * k1[i]; swap d_curr; eval all RHS ---
    for (std::size_t i = 0; i < N; ++i) {
        int n = static_cast<int>(entries_[i].equation->unknown.storedSize);
        eqsys_rk4_tmp<<<(n + 255) / 256, 256>>>(
            phi_tmp_[i]->d_curr,
            entries_[i].equation->unknown.d_curr,
            k1_[i]->d_curr, dt2, n);
        CUDA_CHECK(cudaGetLastError());
    }
    maybeSync_();
    // Apply BCs to phi_tmp (guarded d_curr swap so bc->applyOnGPU works;
    // ScopedDCurrSwap restores the pointer even if a BC throws)
    for (std::size_t i = 0; i < N; ++i) {
        ScopedDCurrSwap swap(*entries_[i].sourceField, phi_tmp_[i]->d_curr);
        entries_[i].batch->applyOnGPU(*entries_[i].sourceField);
    }
    // Point every unknown at its phi_tmp, compute all RHS, restore on exit
    {
        std::vector<ScopedDCurrSwap> swaps;
        swaps.reserve(N);
        for (std::size_t i = 0; i < N; ++i)
            swaps.emplace_back(entries_[i].equation->unknown,
                               phi_tmp_[i]->d_curr);
        for (std::size_t i = 0; i < N; ++i)
            entries_[i].equation->computeRHS(*k2_[i]);
    }

    // --- k3: phi_tmp[i] = phi[i] + dt/2 * k2[i] ---
    for (std::size_t i = 0; i < N; ++i) {
        int n = static_cast<int>(entries_[i].equation->unknown.storedSize);
        eqsys_rk4_tmp<<<(n + 255) / 256, 256>>>(
            phi_tmp_[i]->d_curr,
            entries_[i].equation->unknown.d_curr,
            k2_[i]->d_curr, dt2, n);
        CUDA_CHECK(cudaGetLastError());
    }
    maybeSync_();
    for (std::size_t i = 0; i < N; ++i) {
        ScopedDCurrSwap swap(*entries_[i].sourceField, phi_tmp_[i]->d_curr);
        entries_[i].batch->applyOnGPU(*entries_[i].sourceField);
    }
    {
        std::vector<ScopedDCurrSwap> swaps;
        swaps.reserve(N);
        for (std::size_t i = 0; i < N; ++i)
            swaps.emplace_back(entries_[i].equation->unknown,
                               phi_tmp_[i]->d_curr);
        for (std::size_t i = 0; i < N; ++i)
            entries_[i].equation->computeRHS(*k3_[i]);
    }

    // --- k4: phi_tmp[i] = phi[i] + dt * k3[i] ---
    for (std::size_t i = 0; i < N; ++i) {
        int n = static_cast<int>(entries_[i].equation->unknown.storedSize);
        eqsys_rk4_tmp<<<(n + 255) / 256, 256>>>(
            phi_tmp_[i]->d_curr,
            entries_[i].equation->unknown.d_curr,
            k3_[i]->d_curr, dt, n);
        CUDA_CHECK(cudaGetLastError());
    }
    maybeSync_();
    for (std::size_t i = 0; i < N; ++i) {
        ScopedDCurrSwap swap(*entries_[i].sourceField, phi_tmp_[i]->d_curr);
        entries_[i].batch->applyOnGPU(*entries_[i].sourceField);
    }
    {
        std::vector<ScopedDCurrSwap> swaps;
        swaps.reserve(N);
        for (std::size_t i = 0; i < N; ++i)
            swaps.emplace_back(entries_[i].equation->unknown,
                               phi_tmp_[i]->d_curr);
        for (std::size_t i = 0; i < N; ++i)
            entries_[i].equation->computeRHS(*k4_[i]);
    }

    // --- Final update: phi[i] += dt/6*(k1+2*k2+2*k3+k4) ---
    for (std::size_t i = 0; i < N; ++i) {
        int n = static_cast<int>(entries_[i].equation->unknown.storedSize);
        eqsys_rk4_update<<<(n + 255) / 256, 256>>>(
            entries_[i].equation->unknown.d_curr,
            k1_[i]->d_curr, k2_[i]->d_curr,
            k3_[i]->d_curr, k4_[i]->d_curr,
            dt6, n);
        CUDA_CHECK(cudaGetLastError());
    }
    maybeSync_();

    for (auto& e : entries_)
        e.equation->unknown.advanceTimeLevelGPU();
}

// ===========================================================================
// RK4 advance (CPU)
// ===========================================================================

void EquationSystem::rk4AdvanceCPU_()
{
    const std::size_t N  = entries_.size();
    const double dt2     = dt * 0.5;
    const double dt6     = dt / 6.0;

    // k1
    applyAllBCsCPU_();
    for (std::size_t i = 0; i < N; ++i)
        entries_[i].equation->computeRHSCPU(*k1_[i]);

    // k2: use phi + dt/2 * k1
    // We reuse the rhs_ slot temporarily as k_prev reference isn't needed—
    // just build phi_tmp directly.
    for (std::size_t i = 0; i < N; ++i) {
        auto& phi = entries_[i].equation->unknown.curr;
        auto& tmp = phi_tmp_[i]->curr;
        auto& k   = k1_[i]->curr;
        for (std::size_t j = 0; j < phi.size(); ++j) tmp[j] = phi[j] + dt2 * k[j];
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);
    }
    applyAllBCsCPU_();
    for (std::size_t i = 0; i < N; ++i)
        entries_[i].equation->computeRHSCPU(*k2_[i]);
    for (std::size_t i = 0; i < N; ++i)
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);

    // k3
    for (std::size_t i = 0; i < N; ++i) {
        auto& phi = entries_[i].equation->unknown.curr;
        auto& tmp = phi_tmp_[i]->curr;
        auto& k   = k2_[i]->curr;
        for (std::size_t j = 0; j < phi.size(); ++j) tmp[j] = phi[j] + dt2 * k[j];
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);
    }
    applyAllBCsCPU_();
    for (std::size_t i = 0; i < N; ++i)
        entries_[i].equation->computeRHSCPU(*k3_[i]);
    for (std::size_t i = 0; i < N; ++i)
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);

    // k4
    for (std::size_t i = 0; i < N; ++i) {
        auto& phi = entries_[i].equation->unknown.curr;
        auto& tmp = phi_tmp_[i]->curr;
        auto& k   = k3_[i]->curr;
        for (std::size_t j = 0; j < phi.size(); ++j) tmp[j] = phi[j] + dt * k[j];
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);
    }
    applyAllBCsCPU_();
    for (std::size_t i = 0; i < N; ++i)
        entries_[i].equation->computeRHSCPU(*k4_[i]);
    for (std::size_t i = 0; i < N; ++i)
        std::swap(entries_[i].equation->unknown.curr, phi_tmp_[i]->curr);

    // Final update
    for (std::size_t i = 0; i < N; ++i) {
        auto& phi = entries_[i].equation->unknown.curr;
        auto& k1  = k1_[i]->curr;
        auto& k2  = k2_[i]->curr;
        auto& k3  = k3_[i]->curr;
        auto& k4  = k4_[i]->curr;
        for (std::size_t j = 0; j < phi.size(); ++j)
            phi[j] += dt6 * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
    }

    for (auto& e : entries_)
        e.equation->unknown.advanceTimeLevelCPU();
}

// ===========================================================================
// Public advance / advanceCPU
// ===========================================================================

void EquationSystem::enableAdaptiveDt(const AdaptiveDt& opts)
{
    if (scheme == TimeScheme::RK4)
        throw std::invalid_argument(
            "EquationSystem::enableAdaptiveDt: Euler scheme only "
            "(RK4 needs an embedded error estimate).");
    opts.validate();
    adapt_ = opts;
    adapt_.enabled = true;
}

void EquationSystem::advance()
{
    check::checkPositive(dt, "dt", "EquationSystem::advance");   // public/mutable
    lazyInit_();

    if (scheme == TimeScheme::RK4)
        rk4AdvanceGPU_();
    else
        eulerAdvanceGPU_();

    ++step;
    time += dt;

    if (adapt_.enabled && adapt_.nanCheckEvery > 0
        && step % adapt_.nanCheckEvery == 0) {
        for (auto& e : entries_)
            if (reduce::fieldHasNonFinite(e.equation->unknown))
                throw std::runtime_error(
                    "EquationSystem::advance: NaN/Inf detected in unknown '"
                    + e.equation->unknown.name + "' at step "
                    + std::to_string(step) + " (t=" + std::to_string(time)
                    + ")");
    }

    // Default-on health sentinel (HealthCheck.h) — independent of adaptive dt.
    if (health.due(step))
        for (auto& e : entries_)
            check::checkFieldHealth(e.equation->unknown, step, time,
                                    health.maxAbsLimit);
}

void EquationSystem::advanceCPU()
{
    check::checkPositive(dt, "dt", "EquationSystem::advanceCPU");
    lazyInit_();

    if (scheme == TimeScheme::RK4)
        rk4AdvanceCPU_();
    else
        eulerAdvanceCPU_();

    ++step;
    time += dt;

    // CPU-path sentinel: scan the host copy (authoritative here).
    if (health.due(step))
        for (auto& e : entries_)
            check::checkFieldHealthCPU(e.equation->unknown, step, time,
                                       health.maxAbsLimit);
}

// ===========================================================================
// run
// ===========================================================================

void EquationSystem::run(int nSteps,
                         int callbackEvery,
                         std::function<void(const EquationSystem&)> callback)
{
    for (int s = 0; s < nSteps; ++s) {
        advance();
        if (callback && callbackEvery > 0 && (step % callbackEvery == 0))
            callback(*this);
    }
}

} // namespace PhiX
