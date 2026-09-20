// ---------------------------------------------------------------------------
// module_meshmap — MeshMap: same-domain validation, aligned detection,
// bilinear exactness, conservative closure, block-average fast case,
// direction inference, BC ghost refresh, VectorField loop.
// ---------------------------------------------------------------------------

#include "field/MeshMap.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "core/Error.h"

#include <cmath>
#include <cstdio>
#include <random>

using namespace PhiX;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

template <class E, class Fn>
static bool throwsE(Fn fn) {
    try { fn(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

static void prep(ScalarField& f) {
    f.allocDevice();
    f.uploadAllToDevice();
}

static void test1_validation() {
    printf("[1] construction validation + aligned()\n");
    Mesh a = Mesh::makeUniform2D(CoordSys::CARTESIAN, 50, 0.2, 0.0, 40, 0.25, 0.0);
    Mesh b = Mesh::makeUniform2D(CoordSys::CARTESIAN, 200, 0.05, 0.0, 160, 0.0625, 0.0);
    CHECK([&] { MeshMap m(a, b); return m.aligned(); }(),
          "same domain, 4:1 both axes -> constructs, aligned");

    Mesh c = Mesh::makeUniform2D(CoordSys::CARTESIAN, 75, 0.2 * 50.0 / 75.0, 0.0,
                                 40, 0.25, 0.0);
    CHECK([&] { MeshMap m(a, c); return !m.aligned(); }(),
          "non-integer ratio (1.5:1) -> constructs, NOT aligned");

    Mesh shifted = Mesh::makeUniform2D(CoordSys::CARTESIAN, 50, 0.2, 1.0, 40, 0.25, 0.0);
    CHECK(throwsE<ValidationError>([&] { MeshMap m(a, shifted); }),
          "shifted origin rejected (different domain)");
    Mesh smaller = Mesh::makeUniform2D(CoordSys::CARTESIAN, 50, 0.19, 0.0, 40, 0.25, 0.0);
    CHECK(throwsE<ValidationError>([&] { MeshMap m(a, smaller); }),
          "different extent rejected");
    Mesh cyl = Mesh::makeUniform2D(CoordSys::CYLINDRICAL, 50, 0.2, 0.0, 40, 0.25, 0.0);
    CHECK(throwsE<ValidationError>([&] { MeshMap m(a, cyl); }),
          "non-Cartesian rejected");
    Mesh oneD = Mesh::makeUniform1D(CoordSys::CARTESIAN, 50, 0.2);
    CHECK(throwsE<ValidationError>([&] { MeshMap m(a, oneD); }),
          "dimensionality mismatch rejected");
}

static void test2_bilinear_exactness() {
    printf("[2] interpolate: bilinear exact for linear fields (interior)\n");
    Mesh coarse = Mesh::makeUniform2D(CoordSys::CARTESIAN, 50, 0.2, 0.0, 40, 0.25, 0.0);
    Mesh fine   = Mesh::makeUniform2D(CoordSys::CARTESIAN, 200, 0.05, 0.0, 160, 0.0625, 0.0);
    MeshMap map(coarse, fine);

    auto lin = [](double x, double y, double) { return 3.0 + 2.0 * x - 0.7 * y; };
    ScalarField fc(coarse, "fc", 1), ff(fine, "ff", 1);
    fc.initialize(lin);
    ff.fill(0.0);
    prep(fc); prep(ff);

    map.interpolate(fc, ff);
    ff.downloadCurrFromDevice();

    // interior = fine cells whose sample point is at least one coarse cell
    // from the boundary (edge half-cells are clamp-extrapolated)
    double maxe = 0.0;
    for (int j = 8; j < 160 - 8; ++j)
        for (int i = 8; i < 200 - 8; ++i) {
            const double x = fine.coord(0, i), y = fine.coord(1, j);
            maxe = std::max(maxe,
                            std::fabs(ff.curr[ff.index(i, j)] - lin(x, y, 0)));
        }
    CHECK(maxe < 1e-12, "coarse->fine linear field reproduced exactly (interior)");

    // reverse direction (same map object)
    ScalarField ffl(fine, "ffl", 1), fcb(coarse, "fcb", 1);
    ffl.initialize(lin);
    fcb.fill(0.0);
    prep(ffl); prep(fcb);
    map.interpolate(ffl, fcb);
    fcb.downloadCurrFromDevice();
    maxe = 0.0;
    for (int j = 2; j < 38; ++j)
        for (int i = 2; i < 48; ++i) {
            const double x = coarse.coord(0, i), y = coarse.coord(1, j);
            maxe = std::max(maxe,
                            std::fabs(fcb.curr[fcb.index(i, j)] - lin(x, y, 0)));
        }
    CHECK(maxe < 1e-12, "fine->coarse direction inferred and exact (interior)");
}

static void test3_conservation() {
    printf("[3] conserve: integral closure both directions\n");
    Mesh coarse = Mesh::makeUniform2D(CoordSys::CARTESIAN, 48, 0.25, 0.0, 36, 1.0 / 3.0, 0.0);
    Mesh fine   = Mesh::makeUniform2D(CoordSys::CARTESIAN, 168, 12.0 / 168.0, 0.0,
                                      132, 12.0 / 132.0, 0.0);
    MeshMap map(coarse, fine);
    CHECK(!map.aligned(), "deliberately non-aligned pair");

    ScalarField ffr(fine, "ffr", 1), fcr(coarse, "fcr", 1);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uni(-1.0, 2.0);
    for (int j = 0; j < 132; ++j)
        for (int i = 0; i < 168; ++i)
            ffr.curr[ffr.index(i, j)] = uni(rng);
    fcr.fill(0.0);
    prep(ffr); prep(fcr);

    const double dVf = fine.d[0] * fine.d[1], dVc = coarse.d[0] * coarse.d[1];
    map.conserve(ffr, fcr);
    const double If = reduce::fieldSum(ffr) * dVf;
    const double Ic = reduce::fieldSum(fcr) * dVc;
    CHECK(std::fabs(If - Ic) <= 1e-12 * std::fabs(If),
          "fine->coarse: integral preserved to roundoff (random field)");

    // coarse -> fine
    ScalarField ffb(fine, "ffb", 1);
    ffb.fill(0.0); prep(ffb);
    map.conserve(fcr, ffb);
    const double Ifb = reduce::fieldSum(ffb) * dVf;
    CHECK(std::fabs(Ifb - Ic) <= 1e-12 * std::fabs(Ic),
          "coarse->fine: integral preserved to roundoff");

    // constant stays constant under both operators
    ScalarField cf(fine, "cf", 1), cc(coarse, "cc", 1);
    cf.fill(0.37); cc.fill(0.0);
    prep(cf); prep(cc);
    map.conserve(cf, cc);
    map.interpolate(cf, cc);   // overwrite: also constant
    cc.downloadCurrFromDevice();
    double maxe = 0.0;
    for (int j = 0; j < 36; ++j)
        for (int i = 0; i < 48; ++i)
            maxe = std::max(maxe, std::fabs(cc.curr[cc.index(i, j)] - 0.37));
    CHECK(maxe < 1e-13, "constant field invariant (incl. edges)");
}

static void test4_block_average() {
    printf("[4] aligned 2:1 conserve = exact block average\n");
    Mesh coarse = Mesh::makeUniform2D(CoordSys::CARTESIAN, 16, 0.5, 0.0, 12, 0.5, 0.0);
    Mesh fine   = Mesh::makeUniform2D(CoordSys::CARTESIAN, 32, 0.25, 0.0, 24, 0.25, 0.0);
    MeshMap map(coarse, fine);
    CHECK(map.aligned(), "2:1 pair reported aligned");

    ScalarField ff(fine, "ff", 1), fc(coarse, "fc", 1);
    for (int j = 0; j < 24; ++j)
        for (int i = 0; i < 32; ++i)
            ff.curr[ff.index(i, j)] = std::sin(0.3 * i) + 0.1 * j;
    fc.fill(0.0);
    prep(ff); prep(fc);
    map.conserve(ff, fc);
    fc.downloadCurrFromDevice();

    double maxe = 0.0;
    for (int j = 0; j < 12; ++j)
        for (int i = 0; i < 16; ++i) {
            const double avg = 0.25 *
                (ff.curr[ff.index(2 * i, 2 * j)]     + ff.curr[ff.index(2 * i + 1, 2 * j)]
               + ff.curr[ff.index(2 * i, 2 * j + 1)] + ff.curr[ff.index(2 * i + 1, 2 * j + 1)]);
            maxe = std::max(maxe, std::fabs(fc.curr[fc.index(i, j)] - avg));
        }
    CHECK(maxe < 1e-13, "coarse value equals mean of its four children");
}

static void test5_misuse_and_bc() {
    printf("[5] pair enforcement + BC ghost refresh\n");
    Mesh a = Mesh::makeUniform2D(CoordSys::CARTESIAN, 20, 0.5, 0.0, 20, 0.5, 0.0);
    Mesh b = Mesh::makeUniform2D(CoordSys::CARTESIAN, 40, 0.25, 0.0, 40, 0.25, 0.0);
    Mesh other = Mesh::makeUniform2D(CoordSys::CARTESIAN, 10, 1.0, 0.0, 10, 1.0, 0.0);
    MeshMap map(a, b);

    ScalarField fa(a, "fa", 1), fo(other, "fo", 1);
    prep(fa); prep(fo);
    CHECK(throwsE<ValidationError>([&] { map.interpolate(fa, fo); }),
          "field on a third mesh rejected");

    ScalarField host(a, "host", 1);   // never allocDevice
    ScalarField fb(b, "fb", 1);
    prep(fb);
    CHECK(throwsE<ValidationError>([&] { map.interpolate(host, fb); }),
          "host-only source rejected");

    // BC overload: dst ghosts wrap after transfer
    fa.initialize([](double x, double, double) { return x; });
    fa.uploadAllToDevice();
    PeriodicBC bcx(b.patch("xmin")), bcy(b.patch("ymin"));
    map.interpolate(fa, fb, {&bcx, &bcy});
    fb.downloadCurrFromDevice();
    const double gL = fb.curr[fb.index(-1, 5)];
    const double pR = fb.curr[fb.index(40 - 1, 5)];
    CHECK(gL == pR, "dst ghosts refreshed by BCs after transfer");
}

static void test6_vector_loop() {
    printf("[6] VectorField component loop\n");
    Mesh a = Mesh::makeUniform2D(CoordSys::CARTESIAN, 20, 0.5, 0.0, 20, 0.5, 0.0);
    Mesh b = Mesh::makeUniform2D(CoordSys::CARTESIAN, 40, 0.25, 0.0, 40, 0.25, 0.0);
    MeshMap map(a, b);
    VectorField va(a, "va", 2, 1), vb(b, "vb", 2, 1);
    va[0].fill(1.5); va[1].fill(-2.0);
    va.allocDevice(); va.uploadAllToDevice();
    vb.allocDevice(); vb.uploadAllToDevice();
    map.interpolate(va, vb);
    vb[0].downloadCurrFromDevice();
    vb[1].downloadCurrFromDevice();
    CHECK(vb[0].curr[vb[0].index(7, 7)] == 1.5
              && vb[1].curr[vb[1].index(7, 7)] == -2.0,
          "both components transferred");
    VectorField v3(b, "v3", 3, 1);
    v3.allocDevice(); v3.uploadAllToDevice();
    CHECK(throwsE<ValidationError>([&] { map.interpolate(va, v3); }),
          "component-count mismatch rejected");
}

int main() {
    printf("=== module_meshmap: multi-mesh field transfer ===\n");
    test1_validation();
    test2_bilinear_exactness();
    test3_conservation();
    test4_block_average();
    test5_misuse_and_bc();
    test6_vector_loop();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
