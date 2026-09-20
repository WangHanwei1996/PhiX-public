#pragma once

#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "equation/Term.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// Face-centred operators
//
// These operators bridge cell-centred ScalarField data and face-centred
// FaceField data, enabling conservative finite-volume discretisations.
//
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// interp  —  cell-centre → face-centre linear interpolation
//
// Computes the arithmetic mean of the two neighbouring cell values at each
// interior face:
//
//   f_face[i] = 0.5 * (f_cell[i-1] + f_cell[i])    i = 1..n-1  (interior faces)
//
// Boundary faces (i=0 and i=n) are set to the nearest cell value:
//   f_face[0] = f_cell[0],   f_face[n] = f_cell[n-1]
//
// Operates on CPU by default.  A GPU overload is provided.
// The output FaceField must be constructed for the same mesh and normalAxis.
// ---------------------------------------------------------------------------
void interp(const ScalarField& cell, int axis, FaceField& face);

// GPU version: source field must be on device; result is written to
// face.d_data (which must be allocated before calling).
void interpGPU(const ScalarField& cell, int axis, FaceField& face);

// ---------------------------------------------------------------------------
// faceGrad  —  cell-centre field → face-centre gradient along `axis`
//
// Second-order one-sided / central difference at face i:
//
//   g_face[i] = (f_cell[i] - f_cell[i-1]) / dx    i = 1..n-1  (interior)
//
// Boundary faces (i=0 and i=n) use the ghost cell (must be valid):
//   g_face[0] = (f_cell[0] - f_ghost[-1]) / dx
//   g_face[n] = (f_ghost[n] - f_cell[n-1]) / dx
//
// The output FaceField must be pre-allocated for (mesh, axis).
// ---------------------------------------------------------------------------
void faceGrad(const ScalarField& cell, int axis, FaceField& face);

// GPU version.
void faceGradGPU(const ScalarField& cell, int axis, FaceField& face);

// ---------------------------------------------------------------------------
// divFace  —  conservative divergence from face-centred fluxes → RHS Term
//
// Accumulates:
//   rhs[i,j,k] += coeff * (F_x[i+1,j,k] - F_x[i,j,k]) / dx
//               + coeff * (F_y[i,j+1,k] - F_y[i,j,k]) / dy
//               + coeff * (F_z[i,j,k+1] - F_z[i,j,k]) / dz
//
// Pass only the face fields that are active (1D: flux_x only, etc.).
// Unused pointers should be nullptr.
//
// The returned Term is of type COMPOSITE and is suitable for use in
// Equation::setRHS().
// ---------------------------------------------------------------------------
Term divFace(const FaceField* flux_x,
             const FaceField* flux_y,
             const FaceField* flux_z,
             double coeff = 1.0);

// Convenience 2D overload (no z component):
inline Term divFace(const FaceField& flux_x,
                    const FaceField& flux_y,
                    double coeff = 1.0) {
    return divFace(&flux_x, &flux_y, nullptr, coeff);
}

// Convenience 1D overload:
inline Term divFace(const FaceField& flux_x, double coeff = 1.0) {
    return divFace(&flux_x, nullptr, nullptr, coeff);
}

} // namespace PhiX

// Face-wise point-wise transforms (symmetric to TermPW for cell-centred fields)
#include "operators/FacePW.h"

// Fused face-flux divergence: the faceGrad→interp→facePW→divFace chain
// collapsed into a single kernel (P-4)
#include "operators/DivFaceAssembled.h"
