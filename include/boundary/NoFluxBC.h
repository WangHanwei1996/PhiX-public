#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// NoFluxBC  (zero-gradient / Neumann)
//
// Sets ghost cells on the patch's face equal to the nearest physical
// boundary cell, giving zero normal gradient: d(phi)/dn = 0.
//
// Math (axis X, LOW side, ghost = g):
//   f[-1, j, k] = f[0, j, k]
//   f[-2, j, k] = f[0, j, k]   <- constant extrapolation for all layers
// ---------------------------------------------------------------------------

class NoFluxBC : public BoundaryCondition {
public:
    enum class Closure { Constant, Reflect };
    // Reflect is an even extension about the cell face: f[-g] = f[g-1].
    // It supports wider 2D stencils. Constant preserves the legacy behavior.
    explicit NoFluxBC(const Patch& patch, Closure closure = Closure::Constant);
    Closure closure() const { return closure_; }

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
private:
    Closure closure_;
};

} // namespace PhiX
