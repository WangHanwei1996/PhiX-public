#pragma once

#include "equation/Equation.h"
#include "boundary/BoundaryCondition.h"
#include "boundary/BCBatch.h"
#include "field/ScalarField.h"
#include "solver/Solver.h"      // TimeScheme enum
#include "solver/AdaptiveDt.h"
#include "solver/HealthCheck.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// EquationSystem
//
// Simultaneously evolves a set of coupled equations:
//
//   d(phi_0)/dt = RHS_0(phi_0, phi_1, ..., phi_N)
//   d(phi_1)/dt = RHS_1(phi_0, phi_1, ..., phi_N)
//   ...
//
// KEY DIFFERENCE vs Solver multi-step mode:
//   - multi-step (Solver): equations are applied SEQUENTIALLY.
//     Later equations see already-updated fields from earlier ones.
//     Suitable for operator-split chains (e.g. Cahn-Hilliard mu → c).
//
//   - EquationSystem: all RHS are computed from the SAME time level
//     before any field is updated.  Suitable for fully-coupled systems
//     (e.g. Allen-Cahn multi-phase, reaction-diffusion).
//
// Supported schemes: EULER (forward Euler), RK4 (classical 4th-order).
//
// Workflow:
//   1. Construct with dt and optional time scheme.
//   2. Call add() for each equation.
//   3. Call advance() inside the time loop.
//
// Usage:
//   EquationSystem sys(dt, TimeScheme::EULER);
//   sys.add(eq0, {&bcTop, &bcBot});
//   sys.add(eq_a, {&bcTop, &bcBot});
//   sys.add(eq_b, {&bcTop, &bcBot});
//   for (int s = 0; s < nSteps; ++s) sys.advance();
//
// Notes:
//   - All equations must have device memory allocated before the first advance().
//   - add() may NOT be called after the first advance() (scratch buffers are
//     allocated lazily on the first advance call).
//   - The optional sourceField argument to add() lets the BCs be applied to
//     a field other than equation.unknown (useful when the unknown is a flux
//     whose ghost cells are derived from a different primitive variable).
// ---------------------------------------------------------------------------

class EquationSystem {
public:
    double     dt;
    TimeScheme scheme;
    int        step = 0;
    double     time = 0.0;

    // Per-step NaN/blow-up sentinel over ALL unknowns — ON by default at
    // low cadence (every 100 steps); health.every = 0 disables.
    HealthCheck health;

    explicit EquationSystem(double dt,
                            TimeScheme scheme = TimeScheme::EULER);

    // Non-copyable (owns device scratch memory)
    EquationSystem(const EquationSystem&)            = delete;
    EquationSystem& operator=(const EquationSystem&) = delete;

    // -----------------------------------------------------------------------
    // Register one equation in the system.
    // bcs         : boundary conditions applied to sourceField before each RHS.
    // sourceField : field whose ghost cells to refresh (defaults to equation.unknown).
    // -----------------------------------------------------------------------
    void add(Equation&                       equation,
             std::vector<BoundaryCondition*> bcs         = {},
             ScalarField*                    sourceField = nullptr);

    // -----------------------------------------------------------------------
    // Adaptive time stepping (rate-limited Euler; see AdaptiveDt.h).
    // The controlling rate is max|RHS| over ALL equations — the stiffest
    // one sets dt.  Euler only; throws std::invalid_argument on RK4.
    // After each advance(), `dt` holds the step size actually used.
    // -----------------------------------------------------------------------
    void enableAdaptiveDt(const AdaptiveDt& opts);
    const AdaptiveDt& adaptiveDt() const { return adapt_; }

    // -----------------------------------------------------------------------
    // Advance one time step (GPU path).
    // Lazily allocates scratch buffers on the first call.
    // -----------------------------------------------------------------------
    void advance();

    // CPU fallback (no GPU required; useful for unit tests)
    void advanceCPU();

    // -----------------------------------------------------------------------
    // Run nSteps steps with optional periodic callback.
    // callback receives a const-ref to this EquationSystem after each step.
    // -----------------------------------------------------------------------
    void run(int nSteps,
             int callbackEvery = 0,
             std::function<void(const EquationSystem&)> callback = nullptr);

private:
    struct Entry {
        Equation*                       equation;
        std::vector<BoundaryCondition*> bcs;
        ScalarField*                    sourceField;  // may == equation->unknown
        std::unique_ptr<BCBatch>        batch;        // all bcs in one launch
    };

    std::vector<Entry> entries_;

    bool initialized_ = false;

    // Adaptive-dt controller (enabled == false → fixed dt, zero overhead)
    AdaptiveDt adapt_;

    // One RHS scratch per equation (Euler)
    std::vector<std::unique_ptr<ScalarField>> rhs_;

    // Four stage vectors per equation (RK4)
    std::vector<std::unique_ptr<ScalarField>> k1_, k2_, k3_, k4_;
    // Per-equation phi_tmp (stage-shifted field for RK4 mid-stages)
    std::vector<std::unique_ptr<ScalarField>> phi_tmp_;

    // Allocate all scratch buffers (called once, on first advance)
    void lazyInit_();

    // Full device sync only when some equation uses a non-default stream
    // (default-stream kernels are ordered without it) — v2.18.0.
    void maybeSync_() const;

    // Apply BCs to all source fields
    void applyAllBCsGPU_();
    void applyAllBCsCPU_();

    // Euler advance (GPU/CPU)
    void eulerAdvanceGPU_();
    void eulerAdvanceCPU_();

    // RK4 advance (GPU/CPU)
    void rk4AdvanceGPU_();
    void rk4AdvanceCPU_();
};

} // namespace PhiX
