#pragma once

#include "equation/Equation.h"
#include "boundary/BoundaryCondition.h"
#include "boundary/BCBatch.h"
#include "field/ScalarField.h"
#include "solver/AdaptiveDt.h"
#include "solver/HealthCheck.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// TimeScheme — selectable explicit time integration strategy
// ---------------------------------------------------------------------------
enum class TimeScheme {
    EULER,   // Forward Euler:  phi += dt * rhs
    RK4      // Classical 4th-order Runge-Kutta
};

// ---------------------------------------------------------------------------
// EquationType — how a Solver step updates its unknown field
//
//   TRANSIENT : d(unknown)/dt = RHS  →  unknown += dt * RHS  (time integration)
//   STEADY    : unknown       = RHS  →  unknown  = RHS        (direct assignment)
// ---------------------------------------------------------------------------
enum class EquationType {
    TRANSIENT,
    STEADY
};

// ---------------------------------------------------------------------------
// SolverStep — one equation in a multi-step advance sequence
//
// Before evaluating the equation the solver applies `bcs` to `sourceField`
// (refreshing ghost cells).  `type` controls how the unknown is updated.
//
// Example (Cahn-Hilliard):
//   {&c,  {&bc_x, &bc_y}, &eq_mu, EquationType::STEADY},    // mu = f'(c) - κ∇²c
//   {&mu, {&bc_x, &bc_y}, &eq_c,  EquationType::TRANSIENT}  // dc/dt = M∇²μ
// ---------------------------------------------------------------------------
struct SolverStep {
    ScalarField*                    sourceField;  ///< field whose ghost cells to refresh
    std::vector<BoundaryCondition*> bcs;          ///< BCs applied to sourceField
    Equation*                       equation;     ///< equation to evaluate
    EquationType                    type = EquationType::TRANSIENT;
};

// ---------------------------------------------------------------------------
// Solver
//
// Drives explicit time advancement of one or more Equations.
//
// --- Single-equation mode (original API, unchanged) ---
//   Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::EULER);
//
// --- Multi-step mode (new: STEADY + TRANSIENT equations) ---
//   Solver solver({
//       {&c,  {&bc_x, &bc_y}, &eq_mu, EquationType::STEADY},
//       {&mu, {&bc_x, &bc_y}, &eq_c,  EquationType::TRANSIENT}
//   }, dt);
//   solver.run(nSteps, ...);
//
// Notes:
//   - Multi-step mode currently supports Euler only.
//   - The Solver owns internal scratch fields.
//   - All other objects are non-owning references/pointers.
// ---------------------------------------------------------------------------

class Solver {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    // Single-equation mode
    Solver(Equation&                           equation,
           std::vector<BoundaryCondition*>     bcs,
           double                              dt,
           TimeScheme                          scheme = TimeScheme::EULER);

    // Multi-step mode (ordered STEADY/TRANSIENT equations, Euler only)
    Solver(std::vector<SolverStep>             steps,
           double                              dt,
           TimeScheme                          scheme = TimeScheme::EULER);

    // Non-copyable (owns device scratch memory via Field)
    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    double     dt;            // time step (may be changed between steps)
    TimeScheme scheme;        // integration scheme
    int        step  = 0;     // current step counter (incremented by advance())
    double     time  = 0.0;   // current simulation time

    // Per-step NaN/blow-up sentinel — ON by default at low cadence
    // (every 100 steps); health.every = 0 disables.  See HealthCheck.h.
    HealthCheck health;

    // -----------------------------------------------------------------------
    // Adaptive time stepping (rate-limited Euler; see AdaptiveDt.h).
    // Only supported in single-equation Euler mode — throws
    // std::invalid_argument for RK4 or multi-step solvers.
    // After each advance(), `dt` holds the step size actually used and
    // adaptiveDt().lastMaxRate the controlling max|RHS|.
    // -----------------------------------------------------------------------
    void enableAdaptiveDt(const AdaptiveDt& opts);
    const AdaptiveDt& adaptiveDt() const { return adapt_; }

    // -----------------------------------------------------------------------
    // Single time step (GPU path)
    // -----------------------------------------------------------------------
    void advance();

    // CPU fallback (no GPU required; useful for unit tests)
    void advanceCPU();

    // -----------------------------------------------------------------------
    // Run multiple steps
    //
    // callbackEvery: fire callback every N steps (0 = never)
    // callback:      receives a const ref to this Solver after the step
    // -----------------------------------------------------------------------
    void run(int nSteps,
             int callbackEvery = 0,
             std::function<void(const Solver&)> callback = nullptr);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    const ScalarField& unknown() const { return equation_.unknown; }
    ScalarField&       unknown()       { return equation_.unknown; }

    const Equation&                       equation() const { return equation_; }
    const std::vector<BoundaryCondition*>& bcs()     const { return bcs_; }

private:
    Equation&                       equation_;
    std::vector<BoundaryCondition*> bcs_;
    BCBatch                         bcBatch_;      // all bcs_ in one launch

    // Scratch fields — allocated in constructor, same mesh & ghost as unknown
    ScalarField rhs_;   // used by Euler and each RK4 stage accumulator
    // RK4 stage vectors k1..k4 (only allocated when scheme == RK4)
    ScalarField k1_, k2_, k3_, k4_;
    // RK4 also needs a temporary copy of phi to evaluate mid-stages
    ScalarField phi_tmp_;

    bool use_rk4_   = false;

    // Adaptive-dt controller (enabled == false → fixed dt, zero overhead)
    AdaptiveDt adapt_;

    // Multi-step mode
    bool                                         multiStep_ = false;
    std::vector<SolverStep>                      steps_;
    std::vector<std::unique_ptr<BCBatch>>        stepBatches_;  // per-step BC batch
    std::vector<std::unique_ptr<ScalarField>>    stepScratch_;  // rhs scratch per step (null for STEADY)

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    void applyBCsGPU();
    void applyBCsCPU();

    // Euler: unknown.d_curr += dt * rhs.d_curr  (elementwise GPU kernel)
    void eulerUpdateGPU();
    void eulerUpdateCPU();

    // RK4 sub-steps
    void rk4AdvanceGPU();
    void rk4AdvanceCPU();

    // Multi-step advance (Euler)
    void multiStepAdvanceGPU();
    void multiStepAdvanceCPU();
};

} // namespace PhiX
