#pragma once

// ---------------------------------------------------------------------------
// Interface.h — interface/tip position diagnostics (PFHub BM3 etc.).
//
// interfacePosition() finds the physical coordinate where a field crosses
// `level` along one axis at fixed transverse indices, with linear sub-cell
// interpolation between the bracketing cells.  Only the needed line of
// cells is copied off the device (strided-gather kernel), not the field.
//
//     // dendrite tip along +x through row j0 (φ from solid≈1 to liquid≈0):
//     double xTip = interfacePosition(phi, 0, j0, 0, 0.5, true);
//
// scanFromHigh = true scans from the high-index end towards low and
// returns the FIRST crossing found — the outermost interface (a dendrite
// tip).  false scans from the low end.  Throws std::runtime_error if the
// level is never crossed.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"

namespace PhiX {

double interfacePosition(const ScalarField& f, int axis, int t0, int t1,
                         double level, bool scanFromHigh = true);

} // namespace PhiX
