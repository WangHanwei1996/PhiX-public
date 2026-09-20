#pragma once

// ---------------------------------------------------------------------------
// Anisotropy.h — m-fold surface-energy anisotropy (2D Kobayashi form).
//
// Phase-field dendrite/facet models replace the isotropic gradient energy
// W0²∇φ by W(θ)∇φ with
//
//     a(θ) = 1 + ε·cos(m(θ − θ0)),   θ = atan2(φ_y, φ_x),   W = W0·a
//
// giving the anisotropic driving term (variational derivative):
//
//     ∇·J,   J = W0² [ a²·∇φ  +  a·(da/dθ)·(−φ_y, φ_x) ]
//
// anisoDiv() provides that term as a SINGLE fused kernel: for every cell the
// four face fluxes are evaluated in registers (face-normal gradient +
// averaged tangential gradient, exactly the faceGrad/interp discretisation
// of the classic chain) and their divergence is accumulated into the RHS —
// replacing the hand-built pipeline of ~8 kernel launches and 6 intermediate
// FaceFields (2× faceGrad, 2× interp, 2× facePW, cell-gradient prep, divFace)
// used by the dendrite solver.  Face values are recomputed by both adjacent
// cells (compute is free next to the saved memory traffic) and are formed
// from identical inputs, so the scheme stays CONSERVATIVE (telescoping
// fluxes).
//
// Requirements/limits:
//   • 2D only (m-fold in-plane anisotropy); ghost >= 1 with BCs applied to φ;
//   • the stencil reads DIAGONAL neighbours: at the four domain corners the
//     corner-ghost cells are used, which face-patch BCs do not fill (they are
//     zero after allocDevice).  Keep interfaces away from domain corners, or
//     see BCBatch corner filling (v2.24.0).
//   • ε = 0 reduces exactly to W0²·∇²φ — 5-point for the default CD2
//     scheme, 9-point Patra–Karttunen for AnisoScheme::Iso9 (below).
//
// Usage (dendrite RHS):
//     AnisoParams ap;  ap.W0 = W0;  ap.eps = 0.05;  ap.m = 4;
//     eqPhi.setRHS( anisoDiv(phi, ap) + pw(phi, U, PHIX_FN (...) {...}) );
//
// Cell-centre a(θ) field (for τ(θ) = τ0·a² etc.):
//     anisoFactorOnGPU(phi, aField, ap);   // physical cells, from CD2 ∇φ
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "equation/Term.h"
#include "field/ScalarField.h"

#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Face-normal gradient discretisation of the anisotropic divergence.
//
// CD2 (default): compact two-point normal difference per face.  The ε = 0
//   limit is the 5-point Laplacian, whose leading truncation error carries
//   FOURFOLD mesh symmetry (Fourier: −(kΔx)²·cos4θ/48).  On coarse grids
//   this acts as a spurious grid anisotropy ε_grid ≈ (dx/W)²/48 that
//   competes with the physical ε₄ and can lock dendrite arms onto the mesh
//   diagonals.
//
// Iso9: the compact normal difference is transverse-averaged with weights
//   (1/12, 5/6, 1/12) — i.e. [1 + (Δt²/12)δ²_t]·(f_R − f_L)/Δn.  The ε = 0
//   limit is then EXACTLY the 9-point Patra–Karttunen Laplacian: the
//   fourfold error term vanishes at O(Δx²) (first fourfold at O(Δx⁴)).
//   Still one flux value per face (telescoping-conservative) and
//   checkerboard-damped (symbol −16/3; a two-pass NODAL Iso9 composite has
//   symbol 0 there and blows up — do not confuse the two).  Same ghost = 1.
//   To reach ε_grid ≤ ε₄/5 at ε₄ = 0.007: CD2 needs dx/W < 0.26, Iso9
//   dx/W < 1.0 — ~220× cheaper in 2D explicit stepping (∝ Δx⁻⁴).
// ---------------------------------------------------------------------------
enum class AnisoScheme { CD2, Iso9 };

// "CD2" / "Iso9" (matches the lap()/adv() runtime-string convention);
// throws std::invalid_argument on anything else.
AnisoScheme anisoSchemeFromString(const std::string& name);

struct AnisoParams {
    double W0     = 1.0;   // gradient-energy prefactor (flux carries W0²)
    double eps    = 0.0;   // anisotropy strength ε
    int    m      = 4;     // fold symmetry
    double theta0 = 0.0;   // preferred growth orientation

    // Eggleston convexification (Eggleston, McFadden & Braun, Physica D
    // 150 (2001) 91): beyond the convexity limit ε > 1/(m²−1) the
    // stiffness γ+γ'' turns negative in cones of half-angle θ_m around the
    // γ-maxima (missing orientations, ill-posed evolution).  With
    // regularize = true those cones use the marginally-stable continuation
    //     γ̃(δ) = A·cos(δ),  A = γ(θ_m)/cos(θ_m)   (γ̃+γ̃'' ≡ 0),
    // C¹-matched at δ = ±θ_m via  tan(θ_m) = ε·m·sin(m·θ_m)/γ(θ_m)
    // (solved once on the host).  Below the limit the flag is a no-op.
    bool regularize = false;

    // Face-normal gradient scheme (see AnisoScheme above).  CD2 keeps the
    // legacy path bit-identical; Iso9 removes the O(Δx²) fourfold grid
    // anisotropy.  Honoured by anisoDiv (both overloads) and anisoFactor*
    // (2D); the 3D operator has no scheme knob.
    AnisoScheme scheme = AnisoScheme::CD2;

    // Optional output-cell localization for transformed phase-field models.
    // Where |1-interfacePhi^2| < interfaceCutoff, evaluate the eps=0
    // operator/factor (a=1). This selects the RHS at the receiving cell;
    // it is NOT divergence of a masked flux and need not telescope globally.
    // The phase field must share the input layout and outlive the expression.
    const ScalarField* interfacePhi = nullptr;
    double interfaceCutoff = 1e-6;

    void validate() const;   // throws std::invalid_argument
};

// θ_m and A of the Eggleston continuation for (eps, m); θ_m == 0 (no-op)
// below the convexity limit.  Exposed for testing/diagnostics.
struct AnisoReg { double thetaM; double A; };
AnisoReg anisoComputeRegularization(double eps, int m);

namespace aniso {

// a(θ) — usable inside PHIX_FN functors and kernels.
__host__ __device__ inline Real factor(Real theta, Real eps, int m,
                                       Real theta0) {
    return Real(1) + eps * cos(Real(m) * (theta - theta0));
}

} // namespace aniso

// coeff · ∇·( W0²[a²∇φ + a·a'·(−φ_y, φ_x)] ) as a single fused Term.
Term anisoDiv(const ScalarField& phi, const AnisoParams& p,
              double coeff = 1.0);

// ---------------------------------------------------------------------------
// Per-cell orientation variants (v3.7.0).
//
// theta0Field holds the preferred orientation θ0 of EVERY CELL (radians).
// This is the entry ticket for polycrystals and orientation-field methods:
// a bicrystal is two constant patches of theta0Field selected by a grain
// index, and the frozen-GB method of Tourret & Karma (Acta Mater 82 (2015))
// becomes expressible without hand-rolling the anisotropy in pw functors.
//
// Face orientation convention: a face between two cells takes the θ0 of the
// MORE-SOLID cell (larger φ) — the interface's crystal structure belongs to
// the grain that is growing through that face. Equal values choose the west/south
// endpoint, identically for both receiving cells. With a uniform theta0Field
// this reduces to the scalar-θ0 operator up to the device-vs-host cos/sin
// of the same angle (≤1 ulp; test-enforced at 1e-14).
//
// Requirements: theta0Field on the same mesh/ghost as phi, ghosts VALID
// (refresh its BCs alongside phi's, or fill the full stored array);
// p.theta0 is ignored.  Regularisation (p.regularize) works unchanged.
// ---------------------------------------------------------------------------
Term anisoDiv(const ScalarField& phi, const ScalarField& theta0Field,
              const AnisoParams& p, double coeff = 1.0);

// ===========================================================================
// 3D cubic anisotropy (Karma–Rappel form, axis-aligned crystal axes):
//
//     a(n) = 1 − 3ε + 4ε·(n_x⁴ + n_y⁴ + n_z⁴),   n = ∇φ/|∇φ|
//     J_i  = W0²·a·∂_iφ·[ a + 16ε·(n_i² − S) ],  S = Σ n_k⁴
//
// (exact ∂/∂(∂_iφ) of ½W0²a²|∇φ|²; fully algebraic — no transcendentals.)
// Restricted to a z-invariant field this reduces EXACTLY to the 2D m = 4
// form above with the same ε (test-enforced).  Convexity bound ε ≲ 1/15
// as in 2D m=4; grain rotations (non-axis-aligned crystals) are a future
// extension (rotate ∇φ by Rᵀ, rotate J back).
//
// Requires dim == 3, ghost >= 1.  The stencil reads in-plane diagonal
// neighbours: with BCBatch (v2.24.0 corner pass) the needed EDGE ghosts
// are filled correctly at boundaries.
// ===========================================================================
struct Aniso3DParams {
    double W0  = 1.0;    // gradient-energy prefactor (flux carries W0²)
    double eps = 0.0;    // cubic anisotropy strength ε₄

    // Crystal orientation: R maps LAB gradients into the CRYSTAL frame
    // (row-major 3×3, defaults to identity = axis-aligned).  The flux is
    // evaluated in the crystal frame and rotated back:
    //     J_lab = W0²·a·[ a·p_lab + Rᵀ·v_c ],
    //     v_c,i = 16ε·p_c,i·(n_c,i² − S),   p_c = R·p_lab.
    double R[9] = {1, 0, 0,  0, 1, 0,  0, 0, 1};

    // Bunge z-x-z Euler angles (radians) → R.
    void setEulerZXZ(double phi1, double Phi, double phi2);

    void validate() const;   // throws (also checks R orthonormality);
                             // NOTE: strong-ε convexification is 2D-only
                             // for now — 3D keeps the ε < 0.3 hard bound.
};

namespace aniso {

// a(n) from an (unnormalised) gradient direction — device-friendly.
__host__ __device__ inline Real factor3D(Real px, Real py, Real pz,
                                         Real eps) {
    const Real p2 = px * px + py * py + pz * pz;
    if (p2 <= Real(1e-150)) return Real(1);
    // normalise first — Σp⁴/|p|⁴ under/overflows for extreme gradients
    const Real invp2 = Real(1) / p2;
    const Real nx2 = px * px * invp2;
    const Real ny2 = py * py * invp2;
    const Real nz2 = pz * pz * invp2;
    const Real S = nx2 * nx2 + ny2 * ny2 + nz2 * nz2;
    return Real(1) - Real(3) * eps + Real(4) * eps * S;
}

} // namespace aniso

// coeff · ∇·J with the cubic 3D flux above, as a single fused Term.
Term anisoDiv3D(const ScalarField& phi, const Aniso3DParams& p,
                double coeff = 1.0);

// aOut(physical cells) = a(n(∇φ)) from CD2 cell-centre gradients (3D).
void anisoFactor3DOnGPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p);
void anisoFactor3DOnCPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p);

// aOut(physical cells) = a(θ(∇φ)) from cell-centre gradients (CD2, or the
// 9-point isotropic-error gradient with p.scheme = AnisoScheme::Iso9).
// aOut is pointwise data — no ghost refresh required for pw()-style use.
void anisoFactorOnGPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p);
void anisoFactorOnCPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p);

// Per-cell-orientation variants (v3.7.0): a(θ(∇φ) − theta0Field[c]).
void anisoFactorOnGPU(const ScalarField& phi, const ScalarField& theta0Field,
                      ScalarField& aOut, const AnisoParams& p);
void anisoFactorOnCPU(const ScalarField& phi, const ScalarField& theta0Field,
                      ScalarField& aOut, const AnisoParams& p);

} // namespace PhiX
