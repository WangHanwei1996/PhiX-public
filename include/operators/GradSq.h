#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"
#include "scheme/Isotropic.h"

#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// gradSq(f) — pointwise |grad f|^2 as an RHS Term (stencil, ghost >= 1).
//
// Schemes:
//   CD2  (default): G_{1,0} — squared CD2 differences per axis (1D/2D/3D):
//       |grad f|^2 ~= sum_ax ((f[+1] - f[-1]) / (2 d_ax))^2
//   Iso9 (2D only): G_{2,1} = (2/3) G_{1,0} + (1/3) G_{0,1} — the unique
//       O(h^2)-isotropic 9-point discretization (Ji, Molavi Tabrizi & Karma,
//       J. Comput. Phys. 457 (2022) 111069, Appendix A, Eqs. (66)-(67)):
//       G_{0,1} = [(f_{1,1}-f_{-1,-1})^2 + (f_{-1,1}-f_{1,-1})^2] / (8 dx^2)
//       Leading Fourier error ∝ (k^2)^2 (rotationally invariant); the CD2
//       base has an anisotropic (kx^4 + ky^4) leading error.
//       NOTE: this is a direct discretization of |grad f|^2, NOT the square
//       of the Iso9 gradient — |iso_grad f|^2 is a different (and not
//       isotropic-at-order-h^2) object.
//       RESTRICTION: requires dx == dy (validated at Term construction);
//       falls back to the CD2 formula on 1D/3D meshes.
//
// Motivation: the |grad psi|^2 counter-term of nonlinearly preconditioned
// phase-field equations (phi = tanh(psi/sqrt2)) is a leading O(1)
// differential term and needs an isotropic discretization to suppress
// lattice anisotropy in dendritic growth (Ji 2022, Sec. 2.2); the
// eps4-scaled anisotropy terms do not.
// ---------------------------------------------------------------------------

template<typename Scheme>
Term gradSq(const ScalarField& f, double coeff = 1.0);

// Default (CD2) overload — no second default arg to avoid redefinition
Term gradSq(const ScalarField& f, double coeff);

// Convenience: runtime-string scheme dispatch
// Supported: "CD2" (default), "Iso9"
Term gradSq(const ScalarField& f, const std::string& scheme, double coeff = 1.0);

// Case-driven scheme (scheme/Schemes.h): gradSq(f, sch, c) == gradSq(f, sch.gradSq(f.name), c)
class Schemes;
Term gradSq(const ScalarField& f, const Schemes& sch, double coeff = 1.0);

} // namespace PhiX
