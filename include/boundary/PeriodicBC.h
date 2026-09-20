#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// PeriodicBC
//
// Wraps ghost cells around the opposite physical boundary on the patch's
// axis.  A single PeriodicBC covers BOTH sides of the axis simultaneously
// (the operation is intrinsically symmetric).  Convention: bind the BC to
// the LOW-side patch of the axis; the HIGH-side patch on the same axis
// must also exist and span the same tangential region (typically full face).
//
// Math (axis X, ghost = g):
//   f[-g,     j, k] = f[nx-g, j, k]     (low ghost  copies from far end)
//   f[nx+g-1, j, k] = f[g-1,  j, k]     (high ghost copies from near start)
// ---------------------------------------------------------------------------

class PeriodicBC : public BoundaryCondition {
public:
    explicit PeriodicBC(const Patch& patch);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
};

} // namespace PhiX
