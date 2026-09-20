// ---------------------------------------------------------------------------
// module_movingframe — moving computational window (solver/MovingFrame.h).
//
// 1. SHIFT CONTENT: one X/HIGH shift moves every interior cell exactly one
//    column (new[i] = old[i+1]); slope/constant injection fills the last
//    column from the leading cell's OWN old value.
// 2. FLAT-PAIR TRAP (the §43 runaway): a linear ramp pulled 10 shifts with
//    a slope rule stays EXACTLY linear — every adjacent difference equals
//    the slope, including the last pair.  The wrong extrapolation base
//    (old[last-1]) would leave flat pairs here.
// 3. CONVEYOR: recorded exiting columns reproduce the original data
//    column-by-column; nAcross values per shift; non-recorded field
//    throws on conveyor().
// 4. trackTo: pulls the window until the front position callback drops to
//    the anchor; returns the shift count; runaway guard throws when the
//    callback never satisfies the anchor.
// 5. Axis::Y + Side::LOW mirror: content moves +y, injection at j = 0.
// 6. Validation: layout mismatch and double-add throw.
// ---------------------------------------------------------------------------
#include "solver/MovingFrame.h"
#include "field/ScalarField.h"
#include "field/ReducePW.h"
#include "equation/Term.h"      // PHIX_FN
#include "core/Error.h"
#include "mesh/Mesh.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static int failures = 0;
static void require(bool cond, const std::string& msg) {
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", msg.c_str()); }
    else       {             std::printf("  ok  : %s\n", msg.c_str()); }
}

int main() {
try {
    const int NX = 24, NY = 10;
    const double DX = 0.5;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    NX, DX, 0.0, NY, DX, 0.0);

    auto mkField = [&](const char* n, auto init) {
        ScalarField f(mesh, n, 1);
        f.fillCurr(1e200);   // poisoned ghosts: shifts must not read them
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                f.curr[f.index(i, j, 0)] = init(i, j);
        f.allocDevice(); f.uploadAllToDevice();
        return f;
    };

    // === 1. single shift: content + injection =============================
    {
        ScalarField a = mkField("a", [](int i, int j) {
            return 1.0 * i + 100.0 * j; });
        ScalarField u = mkField("u", [](int i, int j) {
            return std::sin(0.3 * i) + 0.1 * j; });

        MovingFrame frame(mesh, Axis::X, Side::HIGH);
        frame.add(a, InjectRule::slope(+1.0));
        frame.add(u, InjectRule::constant(-1.0));
        frame.shift();
        a.downloadCurrFromDevice(); u.downloadCurrFromDevice();

        double dev = 0.0;
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX - 1; ++i)
                dev = std::max(dev, std::fabs(
                    a.curr[a.index(i, j, 0)] - (1.0 * (i + 1) + 100.0 * j)));
            // injection: old value of the LAST cell itself + slope
            dev = std::max(dev, std::fabs(
                a.curr[a.index(NX - 1, j, 0)]
                - ((NX - 1) + 100.0 * j + 1.0)));
        }
        require(dev == 0.0, "X/HIGH shift: interior + slope injection exact");

        dev = 0.0;
        for (int j = 0; j < NY; ++j)
            dev = std::max(dev,
                std::fabs(u.curr[u.index(NX - 1, j, 0)] - (-1.0)));
        require(dev == 0.0, "constant injection fills the leading column");
        require(frame.shifts() == 1, "shift counter");
    }

    // === 2. flat-pair trap: pulled ramp stays exactly linear ===============
    {
        const double slope = -0.75, b0 = 3.0;
        ScalarField r = mkField("r", [&](int i, int) {
            return slope * i + b0; });
        MovingFrame frame(mesh, Axis::X, Side::HIGH);
        frame.add(r, InjectRule::slope(slope));
        for (int k = 0; k < 10; ++k) frame.shift();
        r.downloadCurrFromDevice();

        double dev = 0.0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX - 1; ++i)
                dev = std::max(dev, std::fabs(
                    (r.curr[r.index(i + 1, j, 0)]
                     - r.curr[r.index(i, j, 0)]) - slope));
        require(dev < 1e-13,
                "10-shift pulled ramp: every adjacent difference == slope "
                "(no flat pairs)");
    }

    // === 3. conveyor ========================================================
    {
        ScalarField c = mkField("c", [](int i, int j) {
            return 10.0 * i + j; });
        ScalarField n = mkField("n", [](int, int) { return 0.0; });
        MovingFrame frame(mesh, Axis::X, Side::HIGH);
        frame.add(c, InjectRule::constant(0.0), /*record=*/true);
        frame.add(n, InjectRule::constant(0.0));
        const int K = 5;
        for (int k = 0; k < K; ++k) frame.shift();

        const auto& conv = frame.conveyor(c);
        require(conv.size() == static_cast<std::size_t>(K) * NY,
                "conveyor holds nAcross values per shift");
        double dev = 0.0;
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < NY; ++j)
                dev = std::max(dev, std::fabs(
                    double(conv[static_cast<std::size_t>(k) * NY + j])
                    - (10.0 * k + j)));
        require(dev == 0.0, "conveyor reproduces the exited columns");

        bool threw = false;
        try { frame.conveyor(n); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "conveyor() of a non-recorded field throws");
    }

    // === 4. trackTo + runaway guard ========================================
    {
        // front = rightmost column where psi > 0; xc = static coordinates
        const int F = 17;
        ScalarField psi = mkField("psi", [&](int i, int) {
            return (i <= F) ? 1.0 : -1.0; });
        ScalarField xc = mkField("xc", [&](int i, int) {
            return (i + 0.5) * DX; });

        MovingFrame frame(mesh, Axis::X, Side::HIGH);
        frame.add(psi, InjectRule::constant(-1.0));   // xc NOT in the frame

        auto front = [&] {
            return reduce::fieldMaxPW(psi, xc, PHIX_FN (Real p, Real x) {
                return p > Real(0) ? x : Real(0); });
        };
        const int T = 11;   // pull until front at column T
        const double anchor = (T + 0.5) * DX;
        const int n = frame.trackTo(anchor, front);
        require(n == F - T, "trackTo pulls exactly front-minus-anchor cells");
        require(std::fabs(front() - anchor) < 1e-12,
                "front sits on the anchor after trackTo");

        // runaway: a position that never drops (constant > anchor)
        bool threw = false;
        try { frame.trackTo(0.0, [] { return 1.0; }); }
        catch (const NumericsError&) { threw = true; }
        require(threw, "runaway guard throws after a domain length");
    }

    // === 5. Axis::Y + Side::LOW mirror ======================================
    {
        ScalarField a = mkField("ay", [](int i, int j) {
            return 1000.0 * i + j; });
        MovingFrame frame(mesh, Axis::Y, Side::LOW);
        frame.add(a, InjectRule::slope(-1.0), /*record=*/true);
        frame.shift();
        a.downloadCurrFromDevice();

        double dev = 0.0;
        for (int i = 0; i < NX; ++i) {
            for (int j = 1; j < NY; ++j)   // content moved +y
                dev = std::max(dev, std::fabs(
                    a.curr[a.index(i, j, 0)] - (1000.0 * i + (j - 1))));
            dev = std::max(dev, std::fabs(
                a.curr[a.index(i, 0, 0)] - (1000.0 * i + 0 - 1.0)));
        }
        require(dev == 0.0, "Y/LOW shift: mirrored content + injection");

        const auto& conv = frame.conveyor(a);
        require(conv.size() == static_cast<std::size_t>(NX), "Y conveyor row");
        double cdev = 0.0;
        for (int i = 0; i < NX; ++i)
            cdev = std::max(cdev, std::fabs(
                double(conv[static_cast<std::size_t>(i)])
                - (1000.0 * i + (NY - 1))));
        require(cdev == 0.0, "Y/LOW conveyor records the j = ny-1 row");
    }

    // === 6. validation ======================================================
    {
        MovingFrame frame(mesh, Axis::X, Side::HIGH);
        ScalarField a = mkField("v1", [](int, int) { return 0.0; });
        frame.add(a, InjectRule::constant(0.0));

        bool threw = false;
        try { frame.add(a, InjectRule::constant(0.0)); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "double-add throws");

        Mesh other = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                         NX + 1, DX, 0.0, NY, DX, 0.0);
        ScalarField b(other, "v2", 1);
        threw = false;
        try { frame.add(b, InjectRule::constant(0.0)); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "wrong-mesh field throws");
    }

    if (failures) { std::printf("module_movingframe: %d FAILURES\n", failures); return 1; }
    std::printf("module_movingframe: all passed\n");
    return 0;
} catch (const std::exception& ex) {
    std::printf("EXCEPTION: %s\n", ex.what());
    return 2;
}
}
