#pragma once

// ---------------------------------------------------------------------------
// GFAThermo.h — shared thermodynamic helpers of the WBM/GFA solver family
// (glass-forming-ability solvers: crystalline φ + amorphous η competition).
//
// Provides, as the single source of truth for NEW solvers:
//   • f_tau(τ)          — amorphous→liquid temperature factor (stage-6
//                         piecewise polynomial; τ = T/T_g)
//   • GFAThermoParams / deltaF_AmToL(T)
//                       — Δf_{am→L}(T) = R·T·ln(1+α)·f(τ)/Vm  [J/m³]
//   • CoolingProtocol   — linear ramp T(t) = T_start − rate·t, clamped to
//                         an explicit [Tmin, Tmax] (usually the table range)
//   • GFATables         — the fL/fS/dfLdc (+ optional McL/McS) CALPHAD
//                         table bundle: load + device upload + views in one
//                         call, plus the T-dependent mobility lookups
//
// Requires nvcc (FreeEnergyTable.h is CUDA-touching).  The pre-v3.3 GFA
// solvers keep their frozen copies of these formulas as regression oracles.
// ---------------------------------------------------------------------------

#include "material/FreeEnergyTable.h"
#include "core/Error.h"

#include <cmath>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// f(τ) — amorphous→liquid temperature factor  [doc/modeling_stage6.md]
// Continuous at τ = 1 (f(1) ≈ −0.109494); → 0⁻ as τ → ∞.
// ---------------------------------------------------------------------------
inline double f_tau(double tau)
{
    if (tau < 1.0) {
        return 1.0
             - 9.9167285e-1  * std::pow(tau, -1.0)
             - 1.11737779e-1 * std::pow(tau,  3.0)
             - 4.96612349e-3 * std::pow(tau,  9.0)
             - 1.11737779e-3 * std::pow(tau, 15.0);
    } else {
        return - 1.05443689e-1 * std::pow(tau,  -5.0)
               - 3.34741816e-3 * std::pow(tau, -15.0)
               - 7.02957924e-4 * std::pow(tau, -25.0);
    }
}

// ---------------------------------------------------------------------------
// Δf_{am→L}(T) = R·T·ln(1+α)·f(T/T_g) / Vm   [J/m³]
// The doc form R·T·ln(1+α)·f(τ) is PER MOLE; dividing by the molar volume
// Vm makes it consistent with f_S − f_L (tables store Gm/Vm in J/m³).
// ---------------------------------------------------------------------------
struct GFAThermoParams {
    double R_gas;   // [J/(mol·K)]
    double alpha;   // [-] driving-force amplitude
    double T_g;     // [K] glass-transition temperature
    double Vm;      // [m³/mol] molar volume (liquid)
};

inline double deltaF_AmToL(double T, const GFAThermoParams& p)
{
    return p.R_gas * T * std::log(1.0 + p.alpha) * f_tau(T / p.T_g) / p.Vm;
}

// ---------------------------------------------------------------------------
// Cooling protocol, clamped to [Tmin, Tmax].  Two modes:
//   LINEAR (default): T(t) = T_start − rate·t — legacy ramp; long runs
//                     saturate at the Tmin clamp.
//   EXP (Newtonian):  T(t) = T_end + (T_start−T_end)·exp(−rate·t/(T_start−T_end))
//                     — bath-temperature approach.  `rate` keeps its meaning
//                     as the INITIAL cooling rate |dT/dt|(0) [K/unit time];
//                     T decays monotonically toward T_end and never crosses
//                     it (log-linear tail — physical for arbitrarily long
//                     runs, no clamp saturation).
// rate = 0 gives an isothermal hold at T_start (clamped) in both modes.
// Aggregate: legacy 4-value brace-init {T_start, rate, Tmin, Tmax} keeps the
// LINEAR behaviour bit-for-bit (defaulted members select it).
// ---------------------------------------------------------------------------
struct CoolingProtocol {
    double T_start;
    double rate;    // [K / unit time]; EXP mode: initial cooling rate
    double Tmin;
    double Tmax;
    bool   expMode = false;   // false = LINEAR (legacy default)
    double T_end   = 0.0;     // EXP only: bath temperature approached as t→∞

    double T(double t) const {
        double v;
        if (!expMode) {
            v = T_start - rate * t;
        } else {
            const double dT = T_start - T_end;
            v = (rate > 0.0 && dT > 0.0)
              ? T_end + dT * std::exp(-rate * t / dT)
              : T_start;
        }
        if (v < Tmin) v = Tmin;
        if (v > Tmax) v = Tmax;
        return v;
    }
};

// ---------------------------------------------------------------------------
// GFATables — the CALPHAD table bundle of the WBM/GFA family.
//
//   fL     f_L(c,T)      liquid free energy            [J/m³]
//   fS     f_S(c,T)      solid  free energy            [J/m³]
//   dfLdc  ∂f_L/∂c(c,T)  analytic derivative table
//   McL/McS (optional)   T-dependent CH mobilities M_c(T) (nc=2 tables,
//                        c-independent lookup); absent → use the constant
//                        fallbacks passed to mcL()/mcS().
//
// load() reads paths from a config section, uploads everything to the
// device, and captures the views — replacing the eight-line boilerplate
// block each GFA solver used to carry.
// ---------------------------------------------------------------------------
class GFATables {
public:
    static GFATables load(const nlohmann::json& tablesCfg) {
        GFATables t;
        t.fL_    = mk(tablesCfg.at("fL").get<std::string>());
        t.fS_    = mk(tablesCfg.at("fS").get<std::string>());
        t.dfLdc_ = mk(tablesCfg.at("dfLdc").get<std::string>());

        const std::string mcL = tablesCfg.value("McL", std::string(""));
        const std::string mcS = tablesCfg.value("McS", std::string(""));
        if (!mcL.empty()) t.McL_ = mk(mcL);
        if (!mcS.empty()) t.McS_ = mk(mcS);
        return t;
    }

    // Device views for RHS functors (capture by value in PHIX_FN lambdas).
    FreeEnergyTableView feL()  const { return fL_->deviceView(); }
    FreeEnergyTableView feS()  const { return fS_->deviceView(); }
    FreeEnergyTableView fdLc() const { return dfLdc_->deviceView(); }

    // T-dependent CH mobilities (host scalars per step): table if given,
    // else the constant fallback.
    double mcL(double T, double fallback) const {
        return McL_ ? McL_->f(0.5, T) : fallback;
    }
    double mcS(double T, double fallback) const {
        return McS_ ? McS_->f(0.5, T) : fallback;
    }
    bool hasMcL() const { return static_cast<bool>(McL_); }
    bool hasMcS() const { return static_cast<bool>(McS_); }

    // Table T-range (from f_L) — the clamp bounds for CoolingProtocol.
    double TMin() const { return fL_->TMin(); }
    double TMax() const { return fL_->TMax(); }

    // Direct access (host-side lookups, diagnostics).
    const FreeEnergyTable& fL()    const { return *fL_; }
    const FreeEnergyTable& fS()    const { return *fS_; }
    const FreeEnergyTable& dfLdc() const { return *dfLdc_; }

private:
    static std::unique_ptr<FreeEnergyTable> mk(const std::string& path) {
        auto p = std::make_unique<FreeEnergyTable>(FreeEnergyTable::fromFile(path));
        p->allocDevice();
        p->uploadToDevice();
        return p;
    }

    std::unique_ptr<FreeEnergyTable> fL_, fS_, dfLdc_, McL_, McS_;
};

} // namespace Material
} // namespace PhiX
