#pragma once

// ---------------------------------------------------------------------------
// SwapGuard.h — RAII pointer swap for RK4 stage evaluation.
//
// The RK4 paths evaluate the RHS "at phi + dt/2 * k" by temporarily pointing
// the unknown's d_curr at a scratch buffer.  A manual save/swap/restore
// leaves the field aliasing scratch memory if computeRHS throws mid-stage
// (later use would double-free or corrupt); ScopedDCurrSwap restores the
// original pointer on scope exit no matter how the scope is left.
//
//   {
//       ScopedDCurrSwap swap(equation.unknown, phi_tmp.d_curr);
//       equation.computeRHS(k2);
//   }   // d_curr restored here, even on throw
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"

namespace PhiX {

class ScopedDCurrSwap {
public:
    ScopedDCurrSwap(ScalarField& f, Real* tmp)
        : f_(f), saved_(f.d_curr) {
        f_.d_curr = tmp;
    }

    ~ScopedDCurrSwap() {
        if (active_) f_.d_curr = saved_;
    }

    ScopedDCurrSwap(const ScopedDCurrSwap&)            = delete;
    ScopedDCurrSwap& operator=(const ScopedDCurrSwap&) = delete;

    // Movable so a std::vector can hold one guard per coupled equation.
    ScopedDCurrSwap(ScopedDCurrSwap&& o) noexcept
        : f_(o.f_), saved_(o.saved_), active_(o.active_) {
        o.active_ = false;
    }
    ScopedDCurrSwap& operator=(ScopedDCurrSwap&&) = delete;

private:
    ScalarField& f_;
    Real*        saved_;
    bool         active_ = true;
};

} // namespace PhiX
