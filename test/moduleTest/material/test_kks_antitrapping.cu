// ---------------------------------------------------------------------------
// module_kks_at — anti-trapping current (material/KKSAntiTrapping.h)
//
// 1. GPU == CPU on a 2D field with a slanted interface and poisoned base
//    flux values (verifies the ACCUMULATE semantics as well).
// 2. 1D sign/magnitude: solidifying front (solid left, moving right) with
//    c_l > c_s must give j_at > 0 (pointing into the liquid) localised at
//    the interface, with peak magnitude ≈ a·W·(c_l−c_s)·max(∂φ/∂t).
// 3. Physics (the reason this module exists): a prescribed interface sweeps
//    a 1D domain at Péclet W·V/D ≈ 0.3 with one-sided mobility
//    (M_s/M_l = 1e-3); the spurious chemical-potential jump measured across
//    the interface must shrink by a large factor when j_at is on, and total
//    solute stays conserved.
// ---------------------------------------------------------------------------

#include "material/KKS.h"
#include "material/KKSAntiTrapping.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/NoFluxBC.h"
#include "operators/FaceOps.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    const double ks = 4.0, cs0 = 0.2;
    const double kl = 1.0, cl0 = 0.7;
    KKSParabolic model(ks, cs0, kl, cl0);

    // =======================================================================
    // 1. GPU == CPU, accumulate semantics (2D, slanted interface)
    // =======================================================================
    {
        const int N = 24;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, 1.0 / N, 0.0, N, 1.0 / N, 0.0);
        const double W = 3.0 / N;

        ScalarField c(mesh, "c", 1), phi(mesh, "phi", 1), dpdt(mesh, "dpdt", 1);
        auto phiFn = [W](double x, double y) {
            return 0.5 * (1.0 - std::tanh((x + 0.4 * y - 0.5) / W));
        };
        for (int j = -1; j <= N; ++j)
        for (int i = -1; i <= N; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j);
            const std::size_t idx = static_cast<std::size_t>(c.index(i, j));
            phi.curr[idx]  = phiFn(x, y);
            c.curr[idx]    = 0.4 + 0.1 * std::sin(7.0 * x) * std::cos(5.0 * y);
            dpdt.curr[idx] = 3.0 * std::exp(-std::pow((x - 0.5) / (2 * W), 2));
        }
        for (ScalarField* f : {&c, &phi, &dpdt}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        KKSAntiTrappingParams at;
        at.W = W;

        FaceField jxG(mesh, 0, "jxG"), jyG(mesh, 1, "jyG");
        FaceField jxC(mesh, 0, "jxC"), jyC(mesh, 1, "jyC");
        jxG.fill(0.7); jyG.fill(-0.3);   // base values: += must preserve them
        jxC.fill(0.7); jyC.fill(-0.3);
        jxG.allocDevice(); jxG.uploadToDevice();
        jyG.allocDevice(); jyG.uploadToDevice();

        kksAddAntiTrappingGPU(model, at, c, phi, dpdt, &jxG, &jyG);
        kksAddAntiTrappingCPU(model, at, c, phi, dpdt, &jxC, &jyC);
        jxG.downloadFromDevice();
        jyG.downloadFromDevice();

        double dev = 0.0;
        for (std::size_t i = 0; i < jxG.storedSize; ++i)
            dev = std::max(dev, std::fabs(jxG.data[i] - jxC.data[i]));
        for (std::size_t i = 0; i < jyG.storedSize; ++i)
            dev = std::max(dev, std::fabs(jyG.data[i] - jyC.data[i]));
        require(dev < 1e-14, "GPU/CPU anti-trapping flux mismatch: "
                             + std::to_string(dev));

        // The interface region must actually have received a contribution
        double changed = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i <= N; ++i)
            changed = std::max(changed, std::fabs(
                jxC.data[static_cast<std::size_t>(jxC.index(i, j))] - 0.7));
        require(changed > 1e-6, "anti-trapping flux is identically zero");
    }

    // =======================================================================
    // 2. 1D sign and magnitude
    // =======================================================================
    {
        const int    N  = 128;
        const double dx = 1.0 / N;
        const double W  = 4 * dx;
        const double V  = 2.0;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

        ScalarField c(mesh, "c", 1), phi(mesh, "phi", 1), dpdt(mesh, "dpdt", 1);
        for (int i = -1; i <= N; ++i) {
            const double x  = mesh.coord(0, i);
            const double s  = (x - 0.5) / W;
            const double p  = 0.5 * (1.0 - std::tanh(s));       // solid left
            const double dp = V / (2.0 * W) / std::cosh(s) / std::cosh(s);
            const std::size_t idx = static_cast<std::size_t>(c.index(i));
            phi.curr[idx]  = p;
            dpdt.curr[idx] = dp;                                 // = −V ∂φ/∂x > 0
            c.curr[idx]    = p * cs0 + (1.0 - p) * cl0;          // eq mixture
        }
        for (ScalarField* f : {&c, &phi, &dpdt}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        KKSAntiTrappingParams at;
        at.W = W;
        FaceField jx(mesh, 0, "jx");
        jx.fill(0.0);
        jx.allocDevice();
        jx.uploadToDevice();

        kksAddAntiTrappingGPU(model, at, c, phi, dpdt, &jx);
        jx.downloadFromDevice();

        // Solidifying, c_l > c_s → the PHYSICAL current points into the
        // liquid (+x); the face field stores −J_at, so values must be <= 0.
        double jmin = 1e300, jmax = -1e300;
        for (int i = 0; i <= N; ++i) {
            const double v = jx.data[static_cast<std::size_t>(jx.index(i))];
            jmin = std::min(jmin, v);
            jmax = std::max(jmax, v);
        }
        require(jmax <= 1e-14,
                "physical anti-trapping current points into the solid");

        // Peak: |∇φ| = |∂φ/∂x| in 1D → |j| = a·W·(c_l−c_s)·(∂φ/∂t) at the
        // face; at the interface centre c is the eq mixture → c_l−c_s ≈
        // cl0−cs0.
        const double expected = at.a * W * (cl0 - cs0) * V / (2.0 * W);
        require(std::fabs(-jmin - expected) < 0.05 * expected,
                "anti-trapping peak magnitude off: " + std::to_string(-jmin)
                + " vs " + std::to_string(expected));
    }

    // =======================================================================
    // 3. Moving interface: trapping suppression + conservation
    // =======================================================================
    {
        const int    N  = 256;
        const double dx = 1.0 / N;
        const double W  = 4 * dx;
        const double V  = 20.0;                  // Pe = V·W/D_l ≈ 0.31
        const double Ml = 1.0, Ms = 1e-3;        // one-sided mobility
        const double x0 = 0.25;
        const double T  = 0.02;                  // front: 0.25 → 0.65
        const double dt = 3e-6;
        const int nSteps = static_cast<int>(T / dt);

        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

        auto runCase = [&](bool withAT, double& dMu, double& massDrift) {
            ScalarField c(mesh, "c", 1), phi(mesh, "phi", 1);
            ScalarField hFr(mesh, "h", 1), dpdt(mesh, "dpdt", 1);
            ScalarField cs(mesh, "cs", 1), cl(mesh, "cl", 1), mu(mesh, "mu", 1);
            FaceField   jx(mesh, 0, "jx"), hFace(mesh, 0, "hFace");

            auto fillPhase = [&](double t) {
                for (int i = -1; i <= N; ++i) {
                    const double x = mesh.coord(0, i);
                    const double s = (x - x0 - V * t) / W;
                    const double p = 0.5 * (1.0 - std::tanh(s));
                    const std::size_t idx = static_cast<std::size_t>(phi.index(i));
                    phi.curr[idx]  = p;
                    hFr.curr[idx]  = static_cast<double>(kks::h(static_cast<Real>(p)));
                    dpdt.curr[idx] = V / (2.0 * W) / std::cosh(s) / std::cosh(s);
                }
                phi.uploadCurrToDevice();
                hFr.uploadCurrToDevice();
                dpdt.uploadCurrToDevice();
            };

            c.initialize([&](double x, double, double) {
                const double p = 0.5 * (1.0 - std::tanh((x - x0) / W));
                const double hh = static_cast<double>(kks::h(static_cast<Real>(p)));
                return hh * cs0 + (1.0 - hh) * cl0;
            });
            for (ScalarField* f : {&c, &phi, &hFr, &dpdt, &cs, &cl, &mu}) {
                f->allocDevice();
                f->uploadAllToDevice();
            }
            jx.allocDevice();
            hFace.allocDevice();

            KKSAntiTrappingParams at;
            at.W = W;

            Equation eqC(c, "solute");
            eqC.setRHS(divFace(jx));
            NoFluxBC bcLo(mesh.facePatch(Axis::X, Side::LOW));
            NoFluxBC bcHi(mesh.facePatch(Axis::X, Side::HIGH));
            NoFluxBC bcMuLo(mesh.facePatch(Axis::X, Side::LOW));
            NoFluxBC bcMuHi(mesh.facePatch(Axis::X, Side::HIGH));
            Solver solver(eqC, {&bcLo, &bcHi}, dt, TimeScheme::EULER);

            const double sum0 = reduce::fieldSum(c);

            for (int s = 0; s < nSteps; ++s) {
                fillPhase(s * dt);
                kksPartitionOnGPU(model, c, hFr, cs, cl, mu);
                bcMuLo.applyOnGPU(mu);
                bcMuHi.applyOnGPU(mu);

                // jx = M(h)·∂μ/∂x on faces  (M = h·Ms + (1−h)·Ml)
                faceGradGPU(mu, 0, jx);
                interpGPU(hFr, 0, hFace);
                facePWGPU(jx, jx, hFace, PHIX_FN (Real gmu, Real hh) {
                    return (hh * Real(1e-3) + (Real(1) - hh) * Real(1.0)) * gmu;
                });
                if (withAT)
                    kksAddAntiTrappingGPU(model, at, c, phi, dpdt, &jx);

                solver.advance();
            }

            massDrift = std::fabs(reduce::fieldSum(c) - sum0)
                        / std::fabs(sum0);

            // μ jump across the interface (front now at x0 + V·T = 0.65)
            kksPartitionOnGPU(model, c, hFr, cs, cl, mu);
            mu.downloadCurrFromDevice();
            const double xf = x0 + V * T;
            const int iBehind = static_cast<int>((xf - 3.0 * W) / dx);
            const int iAhead  = static_cast<int>((xf + 3.0 * W) / dx);
            dMu = mu.curr[static_cast<std::size_t>(mu.index(iBehind))]
                - mu.curr[static_cast<std::size_t>(mu.index(iAhead))];
        };

        double dMuOff, dMuOn, driftOff, driftOn;
        runCase(false, dMuOff, driftOff);
        runCase(true,  dMuOn,  driftOn);

        std::printf("  anti-trapping: dMu(off) = %+.4e  dMu(on) = %+.4e"
                    "  (ratio %.3f)\n", dMuOff, dMuOn,
                    std::fabs(dMuOn / dMuOff));
        std::printf("  mass drift: off %.2e  on %.2e\n", driftOff, driftOn);

        require(driftOff < 1e-9 && driftOn < 1e-9,
                "solute not conserved in moving-interface test");
        require(std::fabs(dMuOff) > 1e-4,
                "baseline run shows no numerical trapping — test is vacuous");
        require(std::fabs(dMuOn) < 0.15 * std::fabs(dMuOff),
                "anti-trapping did not reduce the chemical-potential jump: "
                + std::to_string(dMuOff) + " -> " + std::to_string(dMuOn));
    }

    return 0;
}
