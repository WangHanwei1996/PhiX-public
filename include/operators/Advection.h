#pragma once

// ---------------------------------------------------------------------------
// Advection.h — first-order upwind (donor-cell) advection term.
//
//   adv(u, f)  →  Term representing  coeff · (u · ∇f)
//
// Per axis, the one-sided difference is chosen by the local velocity sign:
//
//   u_x > 0 :  ∂f/∂x ≈ (f[i] - f[i-1]) / dx     (backward / donor cell)
//   u_x < 0 :  ∂f/∂x ≈ (f[i+1] - f[i]) / dx     (forward)
//
// Monotone (no new extrema) and stable for CFL = max|u|·dt/dx < 1, at the
// cost of first-order accuracy (numerical diffusion ~ |u|·dx/2).  Central
// differences applied to advection produce dispersive oscillations — use
// this term whenever a transport velocity enters the model.
//
// Sign convention: adv() is the mathematical term u·∇f.  For the advection
// equation ∂f/∂t + u·∇f = 0 write   eq.setRHS(-1.0 * adv(u, f));
//
// The velocity is read at the cell centre only (no halo needed on u);
// f requires ghost >= 1 with BCs applied as usual.
// ---------------------------------------------------------------------------

#include "equation/Term.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <string>

namespace PhiX {

// coeff · (u · ∇f).  u must have >= mesh.dim components on the same mesh.
// Default scheme: UW1 (1st-order donor cell, ghost 1).
Term adv(const VectorField& u, const ScalarField& f, double coeff = 1.0);

// Scheme-selectable variant:
//   "UW1"   1st-order donor cell            (ghost 1, monotone)
//   "UW2"   2nd-order one-sided upwind      (ghost 2, may over/undershoot
//                                            at discontinuities)
//   "WENO5" 5th-order HJ-WENO (Jiang-Shu)   (ghost 3, essentially
//                                            non-oscillatory, 5th order on
//                                            smooth fields)
//
// Conservation note: all three approximate the DERIVATIVE u·∇f (the
// Hamilton-Jacobi form).  UW1's linear one-sided differences telescope, so
// constant-u periodic transport conserves Σf exactly; UW2/WENO5 do not
// (WENO's nonlinear weights break telescoping) — mass is preserved only to
// truncation accuracy.  For strictly conservative transport put the flux
// through the face-flux chain (FaceOps) instead.
Term adv(const VectorField& u, const ScalarField& f,
         const std::string& schemeName, double coeff = 1.0);

// Case-driven scheme (scheme/Schemes.h): adv(u, f, sch, c) == adv(u, f, sch.adv(f.name), c)
class Schemes;
Term adv(const VectorField& u, const ScalarField& f,
         const Schemes& sch, double coeff = 1.0);

} // namespace PhiX
