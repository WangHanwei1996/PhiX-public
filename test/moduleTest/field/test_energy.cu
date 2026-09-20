// ---------------------------------------------------------------------------
// module_energy — free-energy building blocks (ReducePW.h + fieldGradSq)
//
// 1. fieldSumPW (1/2/3 fields) equals a CPU reference exactly (same values,
//    order-independent up to double reassociation); ghosts poisoned.
// 1b. fieldMaxPW / fieldMinPW (1/2/3 fields) equal CPU references EXACTLY
//    (max/min are reassociation-free); ghosts poisoned with ±1e200 so any
//    ghost leak into the reduction fails loudly.  Includes the masked
//    front-position pattern (max of psi>0 ? x : 0).
// 2. fieldGradSq vs the analytic ∫|∇c|²: c = sin(kx·x)cos(ky·y) on [0,2π]²
//    gives ∫|∇c|² = (kx²+ky²)·π²; the CD2 sum·dV must converge (checked at
//    two resolutions, ratio ~4 = 2nd order).
// 3. End-to-end BM1-style free energy: F = dV·[Σρ(c−cα)²(cβ−c)²
//    + κ/2·Σ|∇c|²] runs and is finite/positive.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "equation/Term.h"     // PHIX_FN
#include "boundary/PeriodicBC.h"
#include "boundary/BCBatch.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // =======================================================================
    // 1. fieldSumPW vs CPU (poisoned ghosts)
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        23, 0.1, 0.0, 17, 0.2, 0.0);
        ScalarField a(mesh, "a", 1), b(mesh, "b", 1), c(mesh, "c", 1);
        for (ScalarField* f : {&a, &b, &c}) f->fillCurr(1e200);
        double r1 = 0.0, r2 = 0.0, r3 = 0.0;
        for (int j = 0; j < 17; ++j)
        for (int i = 0; i < 23; ++i) {
            const double av = std::sin(0.4 * i) + 0.2 * j;
            const double bv = std::cos(0.3 * j) - 0.1 * i;
            const double cv = 0.5 + 0.01 * i * j;
            a.curr[static_cast<std::size_t>(a.index(i, j))] = av;
            b.curr[static_cast<std::size_t>(b.index(i, j))] = bv;
            c.curr[static_cast<std::size_t>(c.index(i, j))] = cv;
            r1 += av * av;
            r2 += av * bv + 1.0;
            r3 += av * bv * cv - av;
        }
        for (ScalarField* f : {&a, &b, &c}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        const double s1 = reduce::fieldSumPW(a,
            PHIX_FN (Real v) { return v * v; });
        const double s2 = reduce::fieldSumPW(a, b,
            PHIX_FN (Real x, Real y) { return x * y + Real(1); });
        const double s3 = reduce::fieldSumPW(a, b, c,
            PHIX_FN (Real x, Real y, Real z) { return x * y * z - x; });

        require(std::fabs(s1 - r1) < 1e-12 * std::max(1.0, std::fabs(r1)),
                "fieldSumPW(1) mismatch");
        require(std::fabs(s2 - r2) < 1e-12 * std::max(1.0, std::fabs(r2)),
                "fieldSumPW(2) mismatch");
        require(std::fabs(s3 - r3) < 1e-12 * std::max(1.0, std::fabs(r3)),
                "fieldSumPW(3) mismatch");
    }

    // =======================================================================
    // 1b. fieldMaxPW / fieldMinPW vs CPU (poisoned ghosts, exact match)
    // =======================================================================
    {
        const int NX = 23, NY = 17;
        const double DX = 0.1;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        NX, DX, 0.0, NY, 0.2, 0.0);
        ScalarField a(mesh, "a", 1), b(mesh, "b", 1), c(mesh, "c", 1);
        ScalarField xc(mesh, "xc", 1);
        a.fillCurr(1e200);  b.fillCurr(-1e200);
        c.fillCurr(1e200);  xc.fillCurr(1e200);
        double mx1 = -1e300, mn1 = 1e300;
        double mx2 = -1e300, mn2 = 1e300;
        double mx3 = -1e300, mn3 = 1e300;
        double frontRef = 0.0;
        for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            const double av = std::sin(0.4 * i) + 0.2 * j;
            const double bv = std::cos(0.3 * j) - 0.1 * i;
            const double cv = 0.5 + 0.01 * i * j;
            const double xv = mesh.coord(0, i);
            a.curr[static_cast<std::size_t>(a.index(i, j))] = av;
            b.curr[static_cast<std::size_t>(b.index(i, j))] = bv;
            c.curr[static_cast<std::size_t>(c.index(i, j))] = cv;
            xc.curr[static_cast<std::size_t>(xc.index(i, j))] = xv;
            mx1 = std::max(mx1, av * av);
            mn1 = std::min(mn1, av * av);
            mx2 = std::max(mx2, av - 2.0 * bv);
            mn2 = std::min(mn2, av - 2.0 * bv);
            mx3 = std::max(mx3, av * bv + cv);
            mn3 = std::min(mn3, av * bv + cv);
            if (av > 0.0) frontRef = std::max(frontRef, xv);
        }
        for (ScalarField* f : {&a, &b, &c, &xc}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        const double gx1 = reduce::fieldMaxPW(a,
            PHIX_FN (Real v) { return v * v; });
        const double gn1 = reduce::fieldMinPW(a,
            PHIX_FN (Real v) { return v * v; });
        const double gx2 = reduce::fieldMaxPW(a, b,
            PHIX_FN (Real x, Real y) { return x - Real(2) * y; });
        const double gn2 = reduce::fieldMinPW(a, b,
            PHIX_FN (Real x, Real y) { return x - Real(2) * y; });
        const double gx3 = reduce::fieldMaxPW(a, b, c,
            PHIX_FN (Real x, Real y, Real z) { return x * y + z; });
        const double gn3 = reduce::fieldMinPW(a, b, c,
            PHIX_FN (Real x, Real y, Real z) { return x * y + z; });

        require(gx1 == mx1, "fieldMaxPW(1) mismatch");
        require(gn1 == mn1, "fieldMinPW(1) mismatch");
        require(gx2 == mx2, "fieldMaxPW(2) mismatch");
        require(gn2 == mn2, "fieldMinPW(2) mismatch");
        require(gx3 == mx3, "fieldMaxPW(3) mismatch");
        require(gn3 == mn3, "fieldMinPW(3) mismatch");

        // masked front-position pattern: max over cells of (a>0 ? x : 0)
        const double front = reduce::fieldMaxPW(a, xc,
            PHIX_FN (Real p, Real x) { return p > Real(0) ? x : Real(0); });
        require(front == frontRef, "masked front-position mismatch");
        std::printf("  max/min PW: front = %.6f (ref %.6f)\n",
                    front, frontRef);
    }

    // =======================================================================
    // 2. fieldGradSq vs analytic (2nd-order convergence)
    // =======================================================================
    {
        const double KX = 2.0, KY = 3.0;
        const double exact = (KX * KX + KY * KY) * M_PI * M_PI;   // ∫|∇c|²
        double err[2];
        int idx = 0;
        for (int N : {64, 128}) {
            const double dx = 2.0 * M_PI / N;
            Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                            N, dx, 0.0, N, dx, 0.0);
            ScalarField f(mesh, "f", 1);
            f.initialize([=](double x, double y, double) {
                return std::sin(KX * x) * std::cos(KY * y);
            });
            f.allocDevice();
            f.uploadAllToDevice();
            PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
            PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
            BCBatch batch;
            batch.build(f, {&bx, &by});
            batch.applyOnGPU(f);

            const double num = reduce::fieldGradSq(f) * dx * dx;
            err[idx++] = std::fabs(num - exact);
        }
        const double order = std::log2(err[0] / err[1]);
        std::printf("  gradSq: err %.3e -> %.3e (order %.2f)\n",
                    err[0], err[1], order);
        require(std::fabs(order - 2.0) < 0.2,
                "fieldGradSq not 2nd-order accurate");
        require(err[1] < 1e-2 * exact, "fieldGradSq error too large");
    }

    // =======================================================================
    // 3. BM1-style free energy composes and is sane
    // =======================================================================
    {
        const int    N  = 96;
        const double L0 = 200.0, dx = L0 / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0);
        ScalarField cf(mesh, "c", 1);
        cf.initialize([](double x, double y, double) {
            return 0.5 + 0.01 * (std::cos(0.105 * x) * std::cos(0.11 * y)
                + std::pow(std::cos(0.13 * x) * std::cos(0.087 * y), 2));
        });
        cf.allocDevice();
        cf.uploadAllToDevice();
        PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
        BCBatch batch;
        batch.build(cf, {&bx, &by});
        batch.applyOnGPU(cf);

        const double kappa = 2.0;
        const double bulk = reduce::fieldSumPW(cf, PHIX_FN (Real v) {
            const Real d1 = v - Real(0.3), d2 = Real(0.7) - v;
            return Real(5.0) * d1 * d1 * d2 * d2;
        });
        const double F = dx * dx * (bulk + 0.5 * kappa
                                    * reduce::fieldGradSq(cf));
        std::printf("  BM1-style F = %.6e\n", F);
        require(std::isfinite(F) && F > 0.0, "free energy not finite/positive");
    }

    return 0;
}
