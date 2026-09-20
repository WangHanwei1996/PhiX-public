#pragma once

// ---------------------------------------------------------------------------
// HealthGuard.h — the health sentinel, detached from Solver/EquationSystem.
//
// The Solver/EquationSystem sentinel (HealthCheck.h), standalone — for
// hand-driven loops stepping raw Equation objects:
//
//     HealthGuard guard;                    // every = 100 by default
//     guard.cfg = {50, 1e6};                // cadence + |max| blow-up limit
//     guard.watch(psi);                     // NaN/Inf (+ maxAbs) scan
//     guard.watch(U);
//     guard.addGuard("front runaway", [&]{ return vFront < 5.0 * Vp; });
//     guard.addGuard("kinetic death", [&]{ return uMax - uMin > 1e-12; });
//     ...
//     for (int step = 0; step < nSteps; ++step) {
//         // ... advance ...
//         guard.check(step, time);          // no-op unless cfg.due(step)
//     }
//
// Semantics:
//   • watch(f): when due, run check::checkFieldHealth(f) — a device NaN/Inf
//     scan (host copy if not device-allocated) plus the optional
//     cfg.maxAbsLimit blow-up gate.
//   • addGuard(name, healthy): `healthy` returns TRUE while the run is fine;
//     the first guard returning false throws NumericsError carrying `name`.
//     Predicates are only evaluated when cfg.due(step) — put reductions in
//     them freely, they run once per cadence, not once per step.
//   • cfg.every = 0 disables everything (same convention as Solver.health).
//
// Fields and predicates are referenced, not owned — keep them alive for the
// guard's lifetime (the usual pattern: everything lives in main()'s scope).
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "core/Error.h"
#include "field/ScalarField.h"
#include "solver/HealthCheck.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace PhiX {

class HealthGuard {
public:
    HealthCheck cfg;   // every / maxAbsLimit — defaults: {100, 0.0}

    void watch(const ScalarField& f) { fields_.push_back(&f); }

    void addGuard(std::string name, std::function<bool()> healthy) {
        guards_.emplace_back(std::move(name), std::move(healthy));
    }

    // Throws NumericsError on the first tripped gate; no-op off-cadence.
    void check(int step, double time) const {
        if (!cfg.due(step)) return;
        force(step, time);
    }

    // Same gates, unconditionally (final-state check, post-mortem, tests).
    void force(int step, double time) const {
        for (const ScalarField* f : fields_)
            check::checkFieldHealth(*f, step, time, cfg.maxAbsLimit);
        for (const auto& g : guards_)
            if (!g.second())
                throw NumericsError("HealthGuard",
                    "guard '" + g.first + "' tripped at step "
                    + std::to_string(step)
                    + " (t=" + std::to_string(time) + ")",
                    "a user-defined health predicate returned false — "
                    "inspect the quantity it monitors");
    }

private:
    std::vector<const ScalarField*> fields_;
    std::vector<std::pair<std::string, std::function<bool()>>> guards_;
};

} // namespace PhiX
