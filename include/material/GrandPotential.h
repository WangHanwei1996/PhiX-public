#pragma once

// ---------------------------------------------------------------------------
// GrandPotential.h — binary grand-potential (μ-primary) model algebra.
//
// The standard Plapp / Choudhury–Nestler binary model with PARABOLIC phase
// free energies of equal curvature k and minima ca (α) / cb (β):
//
//   f_α = ½k(c−ca)²,  f_β = ½k(c−cb)²      ⇒  χ = dc/dμ = 1/k  (constant)
//
//   ∂φ/∂t = M_φ [ κ∇²φ − w g'(φ) + h'(φ)(cb−ca) μ ]
//   ∂μ/∂t = k  [ M∇²μ − (cb−ca) h'(φ) ∂φ/∂t ]
//   c(φ,μ) = ca + h(φ)(cb−ca) + μ/k          (reconstruction, output only)
//
// This header factors the pointwise algebra + the validated parameter pack
// out of the GrandPotential_double-well solver so the GP branch (and its
// future generic-N multicomponent extension) shares one implementation.
// The pre-v3.3 solver keeps its frozen copy as the regression oracle.
//
// The scalar functions are host+device (usable inside PHIX_FN functors);
// the stencil parts (κ∇²φ, M∇²μ) stay in the solver's DSL as lap() terms.
// ---------------------------------------------------------------------------

#include "material/Interpolants.h"
#include "core/Check.h"

#include <nlohmann/json.hpp>

#include <string>

namespace PhiX {
namespace Material {

// --- pointwise algebra ------------------------------------------------------

// dφ/dt bulk part:  M_φ [ −w g'(φ) + h'(φ)((cb−ca)μ − Δb) ]  (κ∇²φ via DSL)
//
// Δb = b_β − b_α is the well-depth difference of the two parabolas — the
// constant part of the grand-potential difference ω_β − ω_α.  It is 0 for
// the symmetric spinodal calibration but nonzero for CALPHAD-fitted phases
// (ParabolicFit.h).  delta_b defaults to 0, and the expression is grouped
// so the delta_b == 0 path is BITWISE identical to the historical form.
PHIX_MAT_HD inline double gp_phi_bulk(double phi, double mu,
                                      double M_phi, double w,
                                      double dcab, bool quintic,
                                      double delta_b = 0.0)
{
    return M_phi * (-w * g_dw_prime(phi)
                    + h_interp_prime(phi, quintic) * dcab * mu
                    - h_interp_prime(phi, quintic) * delta_b);
}

// dμ/dt conservation back-coupling:  −k (cb−ca) h'(φ) ∂φ/∂t   (kM∇²μ via DSL)
PHIX_MAT_HD inline double gp_mu_coupling(double phi, double dphidt,
                                         double k, double dcab, bool quintic)
{
    return -k * dcab * h_interp_prime(phi, quintic) * dphidt;
}

// Composition reconstruction:  c = ca + h(φ)(cb−ca) + μ/k
PHIX_MAT_HD inline double gp_c_reconstruct(double phi, double mu,
                                           double k, double ca,
                                           double dcab, bool quintic)
{
    return ca + h_interp(phi, quintic) * dcab + mu / k;
}

// --- validated parameter pack ----------------------------------------------

struct GPBinaryParams {
    double M;       // solute mobility (μ diffusion)
    double k;       // parabola curvature f''(well)
    double ca, cb;  // phase equilibrium compositions
    double kappa;   // φ gradient-energy coefficient
    double w;       // φ double-well barrier
    double M_phi;   // Allen-Cahn mobility
    bool   quintic; // interpolation family (config "interp": cubic|quintic)

    double chi()  const { return 1.0 / k; }
    double dcab() const { return cb - ca; }

    // Reads the "constants" section; throws ValidationError/ConfigError on
    // missing or unphysical values.
    static GPBinaryParams fromConfig(const nlohmann::json& constants) {
        GPBinaryParams p;
        p.M     = constants.at("M").get<double>();
        p.k     = constants.at("k").get<double>();
        p.ca    = constants.at("ca").get<double>();
        p.cb    = constants.at("cb").get<double>();
        p.kappa = constants.at("kappa").get<double>();
        p.w     = constants.at("w").get<double>();
        p.M_phi = constants.at("M_phi").get<double>();
        p.quintic =
            (constants.value("interp", std::string("cubic")) == "quintic");

        check::checkPositive(p.M,     "M",     "GPBinaryParams");
        check::checkPositive(p.k,     "k",     "GPBinaryParams");
        check::checkPositive(p.kappa, "kappa", "GPBinaryParams");
        check::checkPositive(p.w,     "w",     "GPBinaryParams");
        check::checkPositive(p.M_phi, "M_phi", "GPBinaryParams");
        if (p.ca == p.cb)
            throw ValidationError("GPBinaryParams",
                "ca == cb — the two phase compositions must differ",
                "check the \"constants\" section of the config");
        return p;
    }
};

} // namespace Material
} // namespace PhiX
