// ---------------------------------------------------------------------------
// module_parabolic_fit — material/ParabolicFit.h: exact recovery on
// synthetic parabolic tables, Taylor-at-point variant, non-convex and
// degenerate-window rejection, real CALPHAD table sanity (argv[1] = table
// dir, e.g. data/material_properties/Fe-B).
// ---------------------------------------------------------------------------

#include "material/ParabolicFit.h"
#include "core/Error.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace PhiX;
using namespace PhiX::Material;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

static bool near(double a, double b, double rel) {
    return std::fabs(a - b) <= rel * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

// Synthetic table: f(c,T) = fmin + 0.5*k*(c-ceq)^2 + s*T  (parabola in c,
// linear in T so the T-interpolation is exercised but stays exact).
static FreeEnergyTable makeParabolaTable(double k, double ceq, double fmin,
                                         double s, double cLo, double cHi,
                                         int nc, double TLo, double THi, int nT)
{
    std::vector<double> data(static_cast<std::size_t>(nc) * nT);
    const double dc = (cHi - cLo) / (nc - 1);
    const double dT = (THi - TLo) / (nT - 1);
    for (int ic = 0; ic < nc; ++ic)
        for (int iT = 0; iT < nT; ++iT) {
            const double c = cLo + ic * dc;
            const double T = TLo + iT * dT;
            data[static_cast<std::size_t>(ic) * nT + iT] =
                fmin + 0.5 * k * (c - ceq) * (c - ceq) + s * T;
        }
    return FreeEnergyTable(cLo, cHi, nc, TLo, THi, nT, std::move(data));
}

static void test1_exact_recovery() {
    printf("[1] exact recovery on a synthetic parabola\n");
    const double k = 3.7e9, ceq = 0.412, fmin = -2.5e8, s = 1.0e4;
    FreeEnergyTable tab = makeParabolaTable(k, ceq, fmin, s,
                                            0.0, 1.0, 201, 500.0, 1500.0, 11);
    const double T = 900.0;
    ParabolicPhase p = fitParabola(tab, T, 0.0, 1.0);
    CHECK(near(p.k, k, 1e-9),                 "curvature k recovered");
    CHECK(near(p.c_eq, ceq, 1e-9),            "well position c_eq recovered");
    CHECK(near(p.f_min, fmin + s * T, 1e-9),  "well depth f_min(T) recovered");

    ParabolicPhase q = fitParabola(tab, T, 0.30, 0.55);   // narrow window
    CHECK(near(q.c_eq, ceq, 1e-9), "narrow window finds the same vertex");
}

static void test2_taylor_at_point() {
    printf("[2] fitParabolaAt (Taylor at a given c0)\n");
    const double k = 8.0e8, ceq = 0.333, fmin = 1.0e7;
    FreeEnergyTable tab = makeParabolaTable(k, ceq, fmin, 0.0,
                                            0.0, 1.0, 501, 800.0, 1000.0, 3);
    ParabolicPhase p = fitParabolaAt(tab, 900.0, 0.6);   // off-vertex point
    CHECK(near(p.k, k, 1e-9),      "k from off-vertex expansion");
    CHECK(near(p.c_eq, ceq, 1e-8), "implied vertex matches the true well");
    CHECK(near(p.f_min, fmin, 1e-8), "implied depth matches");

    bool edge = false;
    try { fitParabolaAt(tab, 900.0, 0.0005); }
    catch (const ValidationError&) { edge = true; }
    CHECK(edge, "expansion point at the table edge rejected");
}

static void test3_rejections() {
    printf("[3] non-convex / degenerate-window rejection\n");
    // Concave "table": negative curvature.
    FreeEnergyTable bad = makeParabolaTable(-2.0e9, 0.5, 0.0, 0.0,
                                            0.0, 1.0, 101, 500.0, 600.0, 2);
    bool nonConvex = false;
    try { fitParabola(bad, 550.0, 0.2, 0.8); }
    catch (const NumericsError&) { nonConvex = true; }
    CHECK(nonConvex, "concave landscape raises NumericsError");

    FreeEnergyTable ok = makeParabolaTable(1.0e9, 0.5, 0.0, 0.0,
                                           0.0, 1.0, 101, 500.0, 600.0, 2);
    bool degenerate = false;
    try { fitParabola(ok, 550.0, 0.5001, 0.5002); }   // < one grid cell
    catch (const ValidationError&) { degenerate = true; }
    CHECK(degenerate, "window without an interior node raises ValidationError");
}

static void test4_real_table(const std::string& dir) {
    printf("[4] real CALPHAD table sanity (%s)\n", dir.c_str());
    FreeEnergyTable fL = FreeEnergyTable::fromFile(dir + "/f_L_table.csv");
    const double T = 0.5 * (fL.TMin() + fL.TMax());
    ParabolicPhase p = fitParabola(fL, T, fL.cMin(), fL.cMax());
    CHECK(p.k > 0.0 && std::isfinite(p.k), "liquid fit convex and finite");
    CHECK(p.c_eq > fL.cMin() && p.c_eq < fL.cMax(), "c_eq inside the table range");
    printf("       (f_L @ %.0f K: k=%.3e, c_eq=%.4f)\n", T, p.k, p.c_eq);
}

int main(int argc, char* argv[]) {
    printf("=== module_parabolic_fit: CALPHAD -> parabola bridge ===\n");
    test1_exact_recovery();
    test2_taylor_at_point();
    test3_rejections();
    if (argc > 1) test4_real_table(argv[1]);
    else printf("  SKIP: no table dir argv[1]\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
