// ---------------------------------------------------------------------------
// module_gfa_thermo — material/GFAThermo.h (f_tau / deltaF_AmToL /
// CoolingProtocol / GFATables) + material/GrandPotential.h (algebra +
// validated params).  argv[1] = path to a CALPHAD table directory
// (data/material_properties/Fe-B) for the loader test.
// ---------------------------------------------------------------------------

#include "material/GFAThermo.h"
#include "material/GrandPotential.h"
#include "core/Error.h"

#include <cstdio>
#include <cmath>
#include <string>

using namespace PhiX;
using namespace PhiX::Material;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

// Frozen oracle: f_tau re-typed verbatim from GFA_FeB.cu (pre-v3.3).
static double oracle_f_tau(double tau)
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

static void test1_f_tau() {
    printf("[1] f_tau\n");
    bool ok = true;
    for (double tau : {0.3, 0.5, 0.8, 0.999, 1.001, 1.3, 2.0, 5.0})
        ok = ok && f_tau(tau) == oracle_f_tau(tau);
    CHECK(ok, "bitwise identity with the frozen GFA_FeB copy");

    CHECK(std::fabs(f_tau(1.0 - 1e-9) - f_tau(1.0 + 1e-9)) < 1e-6,
          "continuous across tau = 1 (piecewise junction)");
    CHECK(f_tau(1.0) < -0.10 && f_tau(1.0) > -0.12,
          "f_tau(1) in the expected band (~ -0.1095)");
    CHECK(f_tau(50.0) < 0.0 && f_tau(50.0) > -1e-6,
          "f_tau -> 0^- at high tau (amorphous merges with liquid)");
}

static void test2_deltaF_and_cooling() {
    printf("[2] deltaF_AmToL / CoolingProtocol\n");
    GFAThermoParams tp{8.314, 0.5, 800.0, 5.56e-6};
    const double T = 700.0;
    const double expect =
        8.314 * T * std::log(1.5) * f_tau(T / 800.0) / 5.56e-6;
    CHECK(deltaF_AmToL(T, tp) == expect, "deltaF_AmToL formula exact");

    CoolingProtocol ramp{1800.0, 1e5, 500.0, 2000.0};
    CHECK(ramp.T(0.0) == 1800.0,           "ramp starts at T_start");
    CHECK(ramp.T(1e-3) == 1700.0,          "linear ramp value");
    CHECK(ramp.T(1.0) == 500.0,            "clamps at Tmin");
    CHECK((CoolingProtocol{1800, -1e6, 500, 2000}.T(1.0) == 2000.0),
          "clamps at Tmax (heating branch)");
    CHECK((CoolingProtocol{1800, 0.0, 500, 2000}.T(9.9) == 1800.0),
          "rate = 0 holds isothermal");
}

static void test3_tables(const std::string& dir) {
    printf("[3] GFATables loader (%s)\n", dir.c_str());
    nlohmann::json cfg = {
        {"fL",    dir + "/f_L_table.csv"},
        {"fS",    dir + "/f_S_table.csv"},
        {"dfLdc", dir + "/dfdc_L_table.csv"},
    };
    GFATables t = GFATables::load(cfg);
    CHECK(t.TMin() < t.TMax(), "table T-range sane");

    const double cm = 0.2, Tm = 0.5 * (t.TMin() + t.TMax());
    CHECK(std::isfinite(t.fL().f(cm, Tm)) && std::isfinite(t.fS().f(cm, Tm))
              && std::isfinite(t.dfLdc().f(cm, Tm)),
          "host lookups finite at mid-range");
    CHECK(!t.hasMcL() && t.mcL(Tm, 1.36e-19) == 1.36e-19,
          "absent McL falls back to the constant");

    nlohmann::json cfg2 = cfg;
    cfg2["McL"] = dir + "/M_c_L_table.csv";
    cfg2["McS"] = dir + "/M_c_S_table.csv";
    GFATables t2 = GFATables::load(cfg2);
    CHECK(t2.hasMcL() && t2.hasMcS()
              && std::isfinite(t2.mcL(Tm, 0.0)) && t2.mcL(Tm, 0.0) > 0.0,
          "optional McL/McS tables load and evaluate");

    nlohmann::json bad = cfg;
    bad.erase("fS");
    bool threw = false;
    try { GFATables::load(bad); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "missing required table key throws");
}

static void test4_gp_params_and_algebra() {
    printf("[4] GPBinaryParams / gp_* algebra\n");
    nlohmann::json c = { {"M", 1.0}, {"k", 4.0}, {"ca", 0.3}, {"cb", 0.7},
                         {"kappa", 2.0}, {"w", 1.0}, {"M_phi", 5.0},
                         {"interp", "quintic"} };
    GPBinaryParams p = GPBinaryParams::fromConfig(c);
    CHECK(p.quintic && p.chi() == 0.25 && p.dcab() == (0.7 - 0.3),
          "params parsed; chi/dcab derived");

    nlohmann::json badK = c;  badK["k"] = 0.0;
    CHECK([&] { try { GPBinaryParams::fromConfig(badK); }
                catch (const ValidationError&) { return true; }
                catch (...) {} return false; }(),
          "k = 0 rejected (ValidationError)");
    nlohmann::json badC = c;  badC["cb"] = 0.3;
    CHECK([&] { try { GPBinaryParams::fromConfig(badC); }
                catch (const ValidationError&) { return true; }
                catch (...) {} return false; }(),
          "ca == cb rejected");

    // Algebra spot checks (quintic): at phi=0 both g' and h' vanish.
    CHECK(gp_phi_bulk(0.0, 1.7, p.M_phi, p.w, p.dcab(), true) == 0.0,
          "phi-bulk vanishes in pure phase");
    // c reconstruction at phi=0 / phi=1:
    CHECK(gp_c_reconstruct(0.0, 0.0, p.k, p.ca, p.dcab(), true) == 0.3
              && gp_c_reconstruct(1.0, 0.0, p.k, p.ca, p.dcab(), true) == 0.7,
          "c(phi, mu=0) hits the phase wells");
    // mu back-coupling sign: solidification (dphidt>0) at mid-interface
    // removes solute from the liquid well reference (negative for cb>ca).
    CHECK(gp_mu_coupling(0.5, 1.0, p.k, p.dcab(), true) < 0.0,
          "conservation back-coupling sign");
    // Manual value: -k*dcab*h'(0.5)*d, quintic h'(0.5)=30*0.25*0.25=1.875
    CHECK(gp_mu_coupling(0.5, 2.0, p.k, p.dcab(), true)
              == -p.k * p.dcab() * h_quintic_prime(0.5) * 2.0,
          "conservation back-coupling exact value");
}

int main(int argc, char* argv[]) {
    printf("=== module_gfa_thermo: GFA thermo + GP algebra ===\n");
    test1_f_tau();
    test2_deltaF_and_cooling();
    if (argc > 1) test3_tables(argv[1]);
    else { printf("  SKIP: no table dir argv[1]\n"); }
    test4_gp_params_and_algebra();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
