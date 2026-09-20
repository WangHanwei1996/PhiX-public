#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// FixedBC  (Dirichlet)
//
// Sets all ghost cells on the specified side to a constant value:
//
//   f[-1, j, k] = value          (constant fill, what is IMPLEMENTED)
//
// KNOWN LIMITATION — with cell-centred storage, constant fill enforces the
// boundary-FACE value (f[-1]+f[0])/2 = (value+f[0])/2, NOT phi = value at
// the face; the enforced Dirichlet value is off by (f[0]-value)/2 (a
// first-order boundary-location error).  Exact face enforcement needs
// linear extrapolation f[-1] = 2*value - f[0], planned as an OPT-IN mode
// (changing the default would silently shift every existing Dirichlet
// benchmark).  If your case needs the exact face value today, apply a
// hand-written ghost kernel (see k_apply_xbc in MPF_AC_DW).
// ---------------------------------------------------------------------------

class FixedBC : public BoundaryCondition {
public:
    double value;

    FixedBC(const Patch& patch, double value);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
};

} // namespace PhiX
