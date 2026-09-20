#pragma once

// ---------------------------------------------------------------------------
// AdaptiveDt.h — rate-limited adaptive time stepping for explicit Euler.
//
// Controller idea: for forward Euler the per-step change of the unknown is
// exactly |Δφ| = dt·|RHS|, so once the RHS is computed the largest dt that
// keeps every cell's change below `tol` is known in closed form:
//
//     dt* = safety · tol / max|RHS|          (device reduction, Reduce.h)
//
// The step is then taken with  dt = clamp(min(dt*, dt·grow), dtMin, dtMax).
// No rejection/retry loop is needed — the RHS is evaluated once per step and
// the update simply uses the adapted dt.  Shrinking is immediate (safety
// direction); growth is capped by `grow` per step to avoid oscillation.
//
// Enabled via Solver::enableAdaptiveDt / EquationSystem::enableAdaptiveDt
// (Euler scheme only; RK4 would need an embedded error estimate).
// For an EquationSystem the controlling rate is the max over all equations.
//
// Optional NaN sentinel: every `nanCheckEvery` steps the unknown fields are
// scanned for NaN/±Inf on the device; a hit throws std::runtime_error with
// the step/time, so a diverging run aborts instead of streaming garbage.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <stdexcept>

namespace PhiX {

struct AdaptiveDt {
    double tol    = 0.0;    // max allowed per-cell |Δφ| per step (required, > 0)
    double dtMin  = 0.0;    // hard lower bound on dt (required, > 0)
    double dtMax  = 0.0;    // hard upper bound on dt (required, >= dtMin)
    double grow   = 1.2;    // max dt growth factor per step
    double safety = 0.9;    // fraction of tol actually targeted

    int    nanCheckEvery = 0;   // scan unknowns for NaN/Inf every N steps (0 = off)
                                // SUPERSEDED by Solver::health / EquationSystem::health
                                // (on by default, adaptive-dt independent); kept honoured
                                // for backward compatibility.

    bool   enabled = false;     // set by enableAdaptiveDt()

    double lastMaxRate = 0.0;   // max|RHS| observed at the most recent step

    // dt to use for THIS step, given the current dt and fresh max|RHS|.
    double propose(double dt, double maxRate) const {
        const double target = (maxRate > 0.0) ? safety * tol / maxRate : dtMax;
        return std::max(std::min({target, dt * grow, dtMax}), dtMin);
    }

    void validate() const {
        if (tol <= 0.0)
            throw std::invalid_argument("AdaptiveDt: tol must be > 0");
        if (dtMin <= 0.0 || dtMax < dtMin)
            throw std::invalid_argument(
                "AdaptiveDt: require 0 < dtMin <= dtMax");
        if (grow <= 1.0 || safety <= 0.0 || safety > 1.0)
            throw std::invalid_argument(
                "AdaptiveDt: require grow > 1 and 0 < safety <= 1");
        if (nanCheckEvery < 0)
            throw std::invalid_argument("AdaptiveDt: nanCheckEvery must be >= 0");
    }
};

} // namespace PhiX
