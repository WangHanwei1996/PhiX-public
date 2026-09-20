#pragma once

// ---------------------------------------------------------------------------
// HealthCheck.h — per-step solution-health sentinel configuration.
//
// Both Solver and EquationSystem carry a public `HealthCheck health;` and
// call check::checkFieldHealth on every unknown when due(step): a NaN/Inf
// scan (plus an optional |max| blow-up limit that usually fires BEFORE the
// first NaN) that throws NumericsError with the field name, step and time.
//
// ON BY DEFAULT at a low cadence (every 100 steps) — a diverging run
// aborts with a readable error instead of streaming garbage to disk.
// One GPU reduction per unknown per 100 steps is negligible.
//
//   solver.health.every = 0;              // opt out entirely
//   sys.health = {50, 1e6};               // tighter cadence + blow-up limit
//   sys.health.every = cfg["initialize"].value("nan_check_every", 100);
//
// Decoupled from AdaptiveDt::nanCheckEvery (the old opt-in gate, now
// superseded but still honoured).
// ---------------------------------------------------------------------------

namespace PhiX {

struct HealthCheck {
    int    every       = 100;   // check cadence in steps; 0 disables
    double maxAbsLimit = 0.0;   // extra |max| blow-up guard; 0 = NaN/Inf only

    bool due(int step) const { return every > 0 && step % every == 0; }
};

} // namespace PhiX
