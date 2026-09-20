// ---------------------------------------------------------------------------
// module_elasticity — spectral homogeneous elasticity (ElasticityFFT.h)
//
// 1. Uniform eigenstrain, ⟨σ⟩=0 convention: ε = ε* everywhere → elastic
//    energy 0 (stress-free free-standing cell).
// 2. HAND-DERIVED single-mode anchor: e*(x) = A·cos(kx) (ξ=(k,0) modes):
//    the closed form gives ε11 = (C11+C12)/C11·e*, ε22 = ε12 = 0 —
//    the spectral solution must match to ~1e-12.
// 3. Eshelby uniformity: circular inclusion (isotropic constants,
//    C11−C12 = 2C44): interior total strain is UNIFORM (classic Eshelby
//    property, 2D plane strain).
// 4. Mechanical equilibrium: FD divergence of the reconstructed stress
//    field vanishes to discretisation accuracy (smooth Gaussian e*).
// 5. validate() rejects non-positive-definite constants.
// ---------------------------------------------------------------------------
#include "mechanics/ElasticityFFT.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    const int    N  = 128;
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);
    ElasticParams2D C;
    C.C11 = 250.0; C.C12 = 150.0; C.C44 = 100.0;

    auto mkf = [&](const char* nm) {
        ScalarField f(mesh, nm, 1);
        f.allocDevice();
        return f;
    };

    // ---- 1. uniform eigenstrain: stress-free -----------------------------
    {
        ElasticityFFT2D el(mesh, C);
        ScalarField es = mkf("es"), e11 = mkf("e11"), e22 = mkf("e22"),
                    e12 = mkf("e12"), en = mkf("en");
        es.fill(0.005);
        es.uploadAllToDevice();
        el.solve(es, &e11, &e22, &e12, &en);
        require(std::fabs(reduce::fieldMax(e11) - 0.005) < 1e-12
                && std::fabs(reduce::fieldMin(e11) - 0.005) < 1e-12,
                "uniform e*: e11 != e*");
        require(reduce::fieldMaxAbs(en) < 1e-20,
                "uniform e*: elastic energy not zero");
        std::printf("  [1] uniform e*: stress-free, energy 0\n");
    }

    // ---- 2. hand-derived single-mode anchor ------------------------------
    {
        ElasticityFFT2D el(mesh, C);
        ScalarField es = mkf("es"), e11 = mkf("e11"), e22 = mkf("e22"),
                    e12 = mkf("e12");
        const double A = 0.004, k = 3.0;
        es.initialize([=](double x, double, double) {
            return A * std::cos(k * x);
        });
        es.uploadAllToDevice();
        el.solve(es, &e11, &e22, &e12);

        e11.downloadCurrFromDevice();
        e22.downloadCurrFromDevice();
        e12.downloadCurrFromDevice();
        const double fac = (C.C11 + C.C12) / C.C11;
        double dev = 0.0;
        for (int j = 0; j < N; j += 7)
        for (int i = 0; i < N; ++i) {
            const double ref11 = fac * A * std::cos(k * mesh.coord(0, i));
            const std::size_t idx =
                static_cast<std::size_t>(e11.index(i, j));
            dev = std::max(dev, std::fabs(
                static_cast<double>(e11.curr[idx]) - ref11));
            dev = std::max(dev, std::fabs(
                static_cast<double>(e22.curr[idx])));
            dev = std::max(dev, std::fabs(
                static_cast<double>(e12.curr[idx])));
        }
        std::printf("  [2] single-mode anchor: max dev %.2e\n", dev);
        require(dev < 1e-12,
                "single-mode solution off: " + std::to_string(dev));
    }

    // ---- 3. Eshelby interior uniformity (isotropic constants) ------------
    {
        ElasticParams2D iso;
        iso.C11 = 250.0; iso.C12 = 100.0; iso.C44 = 75.0;  // C11−C12 = 2C44
        ElasticityFFT2D el(mesh, iso);
        ScalarField es = mkf("es"), e11 = mkf("e11"), e22 = mkf("e22"),
                    e12 = mkf("e12");
        const double R = 0.9, w = 0.06, e0 = 0.005;
        es.initialize([=](double x, double y, double) {
            const double r = std::sqrt((x - M_PI) * (x - M_PI)
                                       + (y - M_PI) * (y - M_PI));
            return e0 * 0.5 * (1.0 - std::tanh((r - R) / w));
        });
        es.uploadAllToDevice();
        el.solve(es, &e11, &e22, &e12);

        e11.downloadCurrFromDevice();
        double mn = 1e300, mx = -1e300;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double r = std::sqrt(std::pow(mesh.coord(0, i) - M_PI, 2)
                                       + std::pow(mesh.coord(1, j) - M_PI, 2));
            if (r < 0.5 * R) {
                const double v = e11.curr[static_cast<std::size_t>(
                    e11.index(i, j))];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        std::printf("  [3] Eshelby interior e11: [%.6e, %.6e]"
                    " (spread %.2f%%)\n", mn, mx,
                    100.0 * (mx - mn) / std::fabs(0.5 * (mx + mn)));
        require((mx - mn) < 0.02 * std::fabs(0.5 * (mx + mn)),
                "interior strain not uniform (Eshelby property violated)");
    }

    // ---- 4. equilibrium: FD divergence of stress ≈ 0 ----------------------
    {
        ElasticityFFT2D el(mesh, C);
        ScalarField es = mkf("es"), e11 = mkf("e11"), e22 = mkf("e22"),
                    e12 = mkf("e12");
        es.initialize([=](double x, double y, double) {
            return 0.01 * std::exp(-(std::pow(x - M_PI, 2)
                                     + std::pow(y - M_PI, 2)) / 0.5);
        });
        es.uploadAllToDevice();
        el.solve(es, &e11, &e22, &e12);

        for (ScalarField* f : {&e11, &e22, &e12, &es})
            f->downloadCurrFromDevice();

        auto S = [&](ScalarField& f, int i, int j) {
            return static_cast<double>(f.curr[static_cast<std::size_t>(
                f.index((i + N) % N, (j + N) % N))]);
        };
        auto sig11 = [&](int i, int j) {
            const double d11 = S(e11, i, j) - S(es, i, j);
            const double d22 = S(e22, i, j) - S(es, i, j);
            return C.C11 * d11 + C.C12 * d22;
        };
        auto sig22 = [&](int i, int j) {
            const double d11 = S(e11, i, j) - S(es, i, j);
            const double d22 = S(e22, i, j) - S(es, i, j);
            return C.C12 * d11 + C.C11 * d22;
        };
        auto sig12 = [&](int i, int j) {
            return 2.0 * C.C44 * S(e12, i, j);
        };

        double divMax = 0.0, sigMax = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double dv1 = (sig11(i + 1, j) - sig11(i - 1, j)
                                + sig12(i, j + 1) - sig12(i, j - 1))
                               / (2.0 * dx);
            const double dv2 = (sig12(i + 1, j) - sig12(i - 1, j)
                                + sig22(i, j + 1) - sig22(i, j - 1))
                               / (2.0 * dx);
            divMax = std::max(divMax, std::max(std::fabs(dv1),
                                               std::fabs(dv2)));
            sigMax = std::max(sigMax, std::fabs(sig11(i, j)));
        }
        std::printf("  [4] equilibrium: max|div sigma|·dx / max|sigma|"
                    " = %.2e\n", divMax * dx / sigMax);
        require(divMax * dx / sigMax < 0.05,
                "stress divergence too large (not in equilibrium)");
    }

    // ---- 5. parameter validation ------------------------------------------
    {
        bool threw = false;
        try {
            ElasticParams2D bad;
            bad.C11 = 100.0; bad.C12 = 120.0; bad.C44 = 50.0;  // C12 > C11
            ElasticityFFT2D el(mesh, bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "validate() accepted C12 > C11");
        std::printf("  [5] validate() rejects bad constants\n");
    }

    std::printf("module_elasticity: ALL PASS\n");
    return 0;
}
