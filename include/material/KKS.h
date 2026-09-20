#pragma once

// ---------------------------------------------------------------------------
// KKS.h — Kim-Kim-Suzuki two-phase binary alloy model (PRE 60, 7186 (1999)).
//
// Why KKS: in classical WBM/CH-style models the single concentration field
// inside the diffuse interface feels BOTH phases' free energies, so a
// numerically widened interface produces a non-physical solute excess and a
// spurious chemical contribution to the interface energy.  KKS removes this
// by splitting every cell's concentration into phase concentrations
//
//     c = h·c_s + (1−h)·c_l,        h = h(φ) ∈ [0,1]  (solid fraction)
//
// closed by the EQUAL CHEMICAL POTENTIAL condition
//
//     ∂f_s/∂c_s = ∂f_l/∂c_l = μ.
//
// With parabolic free energies
//
//     f_i(c) = ½ k_i (c − c_i⁰)² + b_i        (i = s, l;  k_i > 0)
//
// the partition is CLOSED FORM (no per-cell Newton):
//
//     μ   = (c − h·c_s⁰ − (1−h)·c_l⁰) / (h/k_s + (1−h)/k_l)
//     c_s = c_s⁰ + μ/k_s,      c_l = c_l⁰ + μ/k_l
//
// Model wiring (the module provides the thermodynamics; the solver owns the
// evolution equations):
//
//   diffusion :  ∂c/∂t = ∇·( M(φ) ∇μ )
//       build the μ field once per step with kksPartitionOnGPU, refresh its
//       ghosts, then use lap(mu, M) or the conservative face-flux chain.
//
//   phase field:  ∂φ/∂t = M_φ [ κ∇²φ − w g'(φ) + h'(φ)·ΔG ]
//       ΔG = f_l(c_l) − f_s(c_s) − μ·(c_l − c_s)   (drivingForce(); > 0
//       favours solidification, = 0 at two-phase equilibrium).  Use
//       KKSView::drivingForce inside a pw2/fpw2 functor — the view is
//       trivially copyable and __host__ __device__.
//
// Interpolant helpers: kks::h(φ) = φ²(3−2φ), kks::dh(φ) = 6φ(1−φ).
// The field-level kernels take a PRE-COMPUTED solid-fraction field so the
// interpolant choice stays with the model; pass φ itself if φ already is
// the fraction.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"

namespace PhiX {

// ===========================================================================
// kks:: interpolant helpers
// ===========================================================================
namespace kks {

__host__ __device__ inline Real h(Real phi) {
    return phi * phi * (Real(3) - Real(2) * phi);
}

__host__ __device__ inline Real dh(Real phi) {
    return Real(6) * phi * (Real(1) - phi);
}

} // namespace kks

// ===========================================================================
// KKSView — trivially-copyable device-side evaluator.
// Obtain from KKSParabolic::view(); safe to capture by value in PHIX_FN
// lambdas and Fused pw nodes.
// ===========================================================================
struct KKSView {
    Real ks, cs0, bs;   // solid : f_s = ½ k_s (c−c_s⁰)² + b_s
    Real kl, cl0, bl;   // liquid: f_l = ½ k_l (c−c_l⁰)² + b_l

    // Chemical potential of the (c, h) mixture under equal-μ partition.
    __host__ __device__ Real mu(Real c, Real hh) const {
        const Real chi = hh / ks + (Real(1) - hh) / kl;   // susceptibility
        return (c - hh * cs0 - (Real(1) - hh) * cl0) / chi;
    }

    __host__ __device__ Real cs(Real c, Real hh) const {
        return cs0 + mu(c, hh) / ks;
    }

    __host__ __device__ Real cl(Real c, Real hh) const {
        return cl0 + mu(c, hh) / kl;
    }

    // All three at once (one χ evaluation).
    __host__ __device__ void partition(Real c, Real hh,
                                       Real& cs_out, Real& cl_out,
                                       Real& mu_out) const {
        mu_out = mu(c, hh);
        cs_out = cs0 + mu_out / ks;
        cl_out = cl0 + mu_out / kl;
    }

    __host__ __device__ Real fs(Real c) const {
        const Real d = c - cs0;
        return Real(0.5) * ks * d * d + bs;
    }

    __host__ __device__ Real fl(Real c) const {
        const Real d = c - cl0;
        return Real(0.5) * kl * d * d + bl;
    }

    // ΔG = f_l(c_l) − f_s(c_s) − μ(c_l − c_s): chemical driving force for
    // solidification; zero at two-phase (common-tangent) equilibrium.
    __host__ __device__ Real drivingForce(Real c, Real hh) const {
        Real csv, clv, muv;
        partition(c, hh, csv, clv, muv);
        return fl(clv) - fs(csv) - muv * (clv - csv);
    }

    // ∂μ/∂c at fixed h — effective inverse susceptibility.  The chemical
    // diffusivity of  ∂c/∂t = ∇·(M∇μ)  is  D_eff = M·dmudc.
    __host__ __device__ Real dmudc(Real hh) const {
        return Real(1) / (hh / ks + (Real(1) - hh) / kl);
    }
};

// ===========================================================================
// KKSParabolic — host-side parameter holder / factory
// ===========================================================================
class KKSParabolic {
public:
    // f_s = ½ ks (c−cs0)² + bs ;  f_l = ½ kl (c−cl0)² + bl
    KKSParabolic(double ks, double cs0, double kl, double cl0,
                 double bs = 0.0, double bl = 0.0);

    KKSView view() const;

    // Two-phase (common-tangent) equilibrium: μ_eq solves
    //   μ²(1/(2k_l) − 1/(2k_s)) + μ(c_l⁰ − c_s⁰) + (b_s − b_l) = 0
    // (equal grand potential).  Of the two roots the one with the smaller
    // |μ| is returned — for b_s = b_l this is μ_eq = 0, i.e. the phase
    // concentrations equal the parabola minima cs0/cl0.
    struct Equilibrium { double mu, cs, cl; };
    Equilibrium equilibrium() const;

    double ks() const { return ks_; }
    double kl() const { return kl_; }
    double cs0() const { return cs0_; }
    double cl0() const { return cl0_; }

private:
    double ks_, cs0_, bs_;
    double kl_, cl0_, bl_;
};

// ===========================================================================
// Field-level partition — one fused kernel filling c_s, c_l and μ from the
// total concentration and a solid-fraction field (h = h(φ), pre-computed by
// the model, or φ itself).  Covers the ENTIRE stored array (ghosts included)
// so the outputs' halos are exactly as consistent as the inputs'.
// All fields must share one layout; outputs must be device-allocated for
// the GPU path.
// ===========================================================================
void kksPartitionOnGPU(const KKSParabolic& model,
                       const ScalarField& c, const ScalarField& hFrac,
                       ScalarField& cs, ScalarField& cl, ScalarField& mu);

void kksPartitionOnCPU(const KKSParabolic& model,
                       const ScalarField& c, const ScalarField& hFrac,
                       ScalarField& cs, ScalarField& cl, ScalarField& mu);

} // namespace PhiX
