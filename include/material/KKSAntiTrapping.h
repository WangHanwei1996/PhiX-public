#pragma once

// ---------------------------------------------------------------------------
// KKSAntiTrapping.h — Karma anti-trapping current for KKS solute transport.
//
// The equal-chemical-potential KKS partition (material/KKS.h) removes the
// STATIC interface artefacts (equilibrium solute excess, chemical interface
// energy).  A MOVING interface at a numerically widened width W still
// generates spurious, W-scaled solute trapping when diffusion is one-sided
// (M_s ≪ M_l): a chemical-potential jump ∝ V·W appears across the interface.
// The anti-trapping current (Karma PRL 87, 115701 (2001); Echebarria, Folch,
// Karma, Plapp PRE 70, 061604 (2004)) cancels that O(W) trapping term:
//
//     ∂c/∂t = ∇·( M(φ) ∇μ − J_at )
//     J_at  = − a · W · (c_l − c_s) · (∂φ/∂t) · ∇φ/|∇φ|     (physical current)
//
// Sign conventions: φ = 1 solid, φ = 0 liquid, h = kks::h(φ).  During
// solidification (∂φ/∂t > 0 in the interface) with c_l > c_s the physical
// current points along −∇φ, i.e. from the solid into the liquid ahead of
// the front, expelling the numerically over-trapped solute.
//
// FACE-FIELD SIGN: PhiX's divFace accumulates  rhs += +∇·F  and the face
// fields of the usual chain hold F = M∇μ, which is the NEGATIVE of the
// physical diffusive current.  This routine therefore accumulates −J_at
// (= +aW(c_l−c_s)(∂φ/∂t)∇φ/|∇φ|) so that a single divFace of the combined
// face field yields ∇·(M∇μ − J_at) as required.
//
// COEFFICIENT `a`: the classical 1/(2√2) belongs to the EFKP convention
// (φ ∈ [−1,1], profile tanh(x/(√2 W))).  Mapping to THIS module's
// convention (φ ∈ [0,1], profile ½(1−tanh(x/W)), h = φ²(3−2φ), mobility
// interpolated in h) rescales it by 2√2 → default a = 1.0.  Numerical
// calibration on a 1D sweeping-interface benchmark (Pe = VW/D ≈ 0.3,
// M_s/M_l = 1e-3) confirms it: the trapping ratio is linear in `a` and
// crosses zero at a ≈ 1.04 (see module_kks_at).  `a` remains exposed —
// re-verify with an interface-width convergence study when you change the
// h(φ) interpolant or the mobility interpolation.  Finite solid
// diffusivity needs the Ohno–Matsuura (PRE 79, 031603 (2009)) correction,
// not included here.
//
// Usage inside the conservative face-flux chain (fluxes accumulated BEFORE
// the single divFace):
//
//     faceGradGPU(mu, 0, jx);                       // jx = ∂μ/∂x on x-faces
//     facePWGPU(jx, jx, Mface_x, PHIX_FN (Real g, Real m) { return m * g; });
//     kksAddAntiTrappingGPU(model, at, c, phi, dphidt, &jx, &jy);
//     eqC.setRHS(divFace(jx, jy));                  // dc/dt = ∇·(M∇μ + j_at)
//
// dphidt: pass the φ-equation RHS field of the CURRENT step (for explicit
// Euler that IS ∂φ/∂t), evaluated before φ is updated.
// c, phi, dphidt need refreshed ghosts (BCs applied) on entry.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "material/KKS.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"

namespace PhiX {

struct KKSAntiTrappingParams {
    double W;                          // interface width parameter of the φ eq
    double a = 1.0;                    // convention-mapped 1/(2√2); see header
    double gradEps = 1e-10;            // |∇φ| below this → no interface → zero flux

    void validate() const;             // throws std::invalid_argument
};

// Accumulate (+=) j_at onto the active face-flux components.  Pass exactly
// the face fields of the mesh's active axes (jy/jz nullptr when inactive).
void kksAddAntiTrappingGPU(const KKSParabolic& model,
                           const KKSAntiTrappingParams& p,
                           const ScalarField& c,
                           const ScalarField& phi,
                           const ScalarField& dphidt,
                           FaceField* jx,
                           FaceField* jy = nullptr,
                           FaceField* jz = nullptr);

// CPU mirror (host arrays; for tests / no-GPU runs).
void kksAddAntiTrappingCPU(const KKSParabolic& model,
                           const KKSAntiTrappingParams& p,
                           const ScalarField& c,
                           const ScalarField& phi,
                           const ScalarField& dphidt,
                           FaceField* jx,
                           FaceField* jy = nullptr,
                           FaceField* jz = nullptr);

} // namespace PhiX
