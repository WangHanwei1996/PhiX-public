// ---------------------------------------------------------------------------
// module_gradsq — gradSq(f): |grad f|^2 Term, CD2 (G_{1,0}) and Iso9 (G_{2,1})
//
// 1. Polynomial exactness (both schemes, GPU):
//    a) linear  f = 2x + 3y      -> |grad f|^2 = 13 exactly (+ coeff scaling)
//    b) quadratic f = x^2 - y^2  -> 4(x^2 + y^2) exactly (CD2 differences are
//       exact on quadratics; the diagonal base reproduces it too)
// 2. Plane-wave isotropy (the point of Iso9): f = sin(k.r) with analytically
//    filled ghosts, |k| = 4 at 0 deg and 45 deg.  Leading errors:
//      CD2  : (kx^4 + ky^4) h^2/3  -> E(0)/E(45) ~= 2       (anisotropic)
//      Iso9 : (k^2)^2 h^2/3        -> E(0) ~= E(45) within ~1% (isotropic)
//    Assert Iso9 orientation spread < 5%, CD2 spread > 40%.
// 3. Grid convergence: n=64 -> n=128 on k=(3,2), error ratio ~= 4 (order 2),
//    both schemes.
// 4. GPU vs CPU bitwise-level agreement (<= 1e-12) on a smooth field.
// 5. String dispatch: "Iso9" works, unknown scheme throws; Iso9 on a
//    non-square mesh (dx != dy) throws ValidationError.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "operators/GradSq.h"
#include "field/ScalarField.h"
#include "core/Error.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Fill curr at ALL stored cells (physical + ghost) from fn(x, y).
template<typename Fn>
static void fillWithGhost2D(ScalarField& f, Fn fn) {
    const int g = f.ghost;
    for (int j = -g; j < f.mesh.n[1] + g; ++j)
    for (int i = -g; i < f.mesh.n[0] + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            fn(f.mesh.coord(0, i), f.mesh.coord(1, j));
}

// Max |rhs - ref(x,y)| over physical cells, GPU path.
template<typename Ref>
static double maxErrGPU(ScalarField& src, const Term& term, Ref ref) {
    Equation eq(src, "gradsq_test");
    eq.setRHS(term);
    ScalarField rhs(src.mesh, "rhs", src.ghost);
    rhs.allocDevice();
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();
    double err = 0.0;
    for (int j = 0; j < src.mesh.n[1]; ++j)
    for (int i = 0; i < src.mesh.n[0]; ++i)
        err = std::max(err, std::fabs(
            rhs.curr[static_cast<std::size_t>(rhs.index(i, j))]
            - ref(src.mesh.coord(0, i), src.mesh.coord(1, j))));
    return err;
}

// NOTE: ScalarField stores `const Mesh&` — every Mesh must outlive its fields,
// so meshes are constructed in the scope that owns the fields.
static Mesh makeMesh(int n, double L) {
    return Mesh::makeUniform2D(CoordSys::CARTESIAN,
                               n, L / n, 0.0, n, L / n, 0.0);
}

// Plane-wave error of a scheme at wavevector (kx, ky), GPU path.
static double planeWaveErr(int n, double kx, double ky,
                           const std::string& scheme) {
    Mesh mesh = makeMesh(n, 2.0 * M_PI);
    ScalarField f(mesh, "f", 1);
    fillWithGhost2D(f, [=](double x, double y) {
        return std::sin(kx * x + ky * y);
    });
    f.allocDevice();
    f.uploadAllToDevice();
    const double k2 = kx * kx + ky * ky;
    return maxErrGPU(f, gradSq(f, scheme), [=](double x, double y) {
        const double c = std::cos(kx * x + ky * y);
        return k2 * c * c;
    });
}

int main() try {
    // =======================================================================
    // 1. Polynomial exactness
    // =======================================================================
    for (const std::string scheme : {"CD2", "Iso9"}) {
        Mesh mesh = makeMesh(48, 4.8);
        ScalarField f(mesh, "f", 1);
        fillWithGhost2D(f, [](double x, double y) { return 2.0 * x + 3.0 * y; });
        f.allocDevice();
        f.uploadAllToDevice();
        double e = maxErrGPU(f, gradSq(f, scheme),
                             [](double, double) { return 13.0; });
        require(e < 1e-11, scheme + ": linear field not exact, err=" + std::to_string(e));

        double e2 = maxErrGPU(f, gradSq(f, scheme, 2.5),
                              [](double, double) { return 32.5; });
        require(e2 < 1e-11, scheme + ": coeff scaling broken");

        ScalarField q(mesh, "q", 1);
        fillWithGhost2D(q, [](double x, double y) { return x * x - y * y; });
        q.allocDevice();
        q.uploadAllToDevice();
        double e3 = maxErrGPU(q, gradSq(q, scheme), [](double x, double y) {
            return 4.0 * (x * x + y * y);
        });
        require(e3 < 1e-9, scheme + ": quadratic field not exact, err=" + std::to_string(e3));
    }
    std::cout << "[1] polynomial exactness (CD2 + Iso9)      OK\n";

    // =======================================================================
    // 2. Plane-wave isotropy at |k| = 4, n = 128
    // =======================================================================
    {
        const double q = 4.0, r = q / std::sqrt(2.0);
        double cd0  = planeWaveErr(128, q, 0.0, "CD2");
        double cd45 = planeWaveErr(128, r, r,   "CD2");
        double is0  = planeWaveErr(128, q, 0.0, "Iso9");
        double is45 = planeWaveErr(128, r, r,   "Iso9");
        std::cout << "    CD2 : E(0)=" << cd0 << "  E(45)=" << cd45
                  << "  spread=" << std::fabs(cd0 - cd45) / cd0 << "\n";
        std::cout << "    Iso9: E(0)=" << is0 << "  E(45)=" << is45
                  << "  spread=" << std::fabs(is0 - is45) / is0 << "\n";
        require(std::fabs(is0 - is45) / is0 < 0.05,
                "Iso9 orientation spread >= 5% (not isotropic)");
        require(std::fabs(cd0 - cd45) / cd0 > 0.40,
                "CD2 orientation spread < 40% (test not discriminating)");
    }
    std::cout << "[2] plane-wave isotropy (Iso9 < 5% spread) OK\n";

    // =======================================================================
    // 3. Grid convergence, k = (3, 2): expect O(h^2)
    // =======================================================================
    for (const std::string scheme : {"CD2", "Iso9"}) {
        double eC = planeWaveErr(64,  3.0, 2.0, scheme);
        double eF = planeWaveErr(128, 3.0, 2.0, scheme);
        double ratio = eC / eF;
        require(ratio > 3.0 && ratio < 5.0,
                scheme + ": convergence ratio " + std::to_string(ratio)
                + " not ~4 (order 2)");
    }
    std::cout << "[3] O(h^2) convergence (CD2 + Iso9)        OK\n";

    // =======================================================================
    // 4. GPU vs CPU agreement
    // =======================================================================
    for (const std::string scheme : {"CD2", "Iso9"}) {
        Mesh mesh = makeMesh(64, 2.0 * M_PI);
        ScalarField f(mesh, "f", 1);
        fillWithGhost2D(f, [](double x, double y) {
            return std::sin(3.0 * x) * std::cos(2.0 * y) + 0.3 * std::sin(x + 5.0 * y);
        });
        f.allocDevice();
        f.uploadAllToDevice();

        Equation eqG(f, "g");
        eqG.setRHS(gradSq(f, scheme));
        ScalarField rG(f.mesh, "rG", 1);
        rG.allocDevice();
        eqG.computeRHS(rG);
        rG.downloadCurrFromDevice();

        Equation eqC(f, "c");
        eqC.setRHS(gradSq(f, scheme));
        ScalarField rC(f.mesh, "rC", 1);
        eqC.computeRHSCPU(rC);

        double d = 0.0;
        for (int j = 0; j < 64; ++j)
        for (int i = 0; i < 64; ++i)
            d = std::max(d, std::fabs(
                rG.curr[static_cast<std::size_t>(rG.index(i, j))]
                - rC.curr[static_cast<std::size_t>(rC.index(i, j))]));
        require(d <= 1e-12, scheme + ": GPU/CPU mismatch " + std::to_string(d));
    }
    std::cout << "[4] GPU vs CPU (<= 1e-12)                  OK\n";

    // =======================================================================
    // 5. Dispatch and validation
    // =======================================================================
    {
        Mesh mesh = makeMesh(16, 1.6);
        ScalarField f(mesh, "f", 1);
        bool threw = false;
        try { gradSq(f, "CD4"); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "unknown scheme did not throw invalid_argument");

        Mesh rect = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        16, 0.1, 0.0, 16, 0.2, 0.0);
        ScalarField g(rect, "g", 1);
        threw = false;
        try { gradSq(g, "Iso9"); } catch (const ValidationError&) { threw = true; }
        require(threw, "Iso9 on dx != dy did not throw ValidationError");
        gradSq(g, "CD2");    // CD2 on non-square mesh must stay legal
    }
    std::cout << "[5] dispatch + dx==dy validation           OK\n";

    std::cout << "module_gradsq: ALL PASSED\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "module_gradsq FAILED: " << e.what() << "\n";
    return 1;
}
