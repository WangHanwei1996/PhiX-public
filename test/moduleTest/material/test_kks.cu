// ---------------------------------------------------------------------------
// module_kks — Kim-Kim-Suzuki two-phase partition (material/KKS.h)
//
// 1. Closed-form partition identities at scattered (c, h) points:
//    equal chemical potential, mixture consistency c = h·cs + (1−h)·cl,
//    single-phase limits, interpolant endpoints.
// 2. equilibrium(): symmetric case (bs == bl) gives μ_eq = 0 and the
//    parabola minima; an equilibrium-consistent profile has ΔG ≈ 0 for
//    every h.
// 3. GPU field kernel == CPU reference bitwise-tolerance check.
// 4. 1D physics relaxation (THE KKS selling point): a static tanh interface,
//    off-equilibrium initial c, evolved by dc/dt = ∇²μ with no-flux walls:
//      • total solute conserved,
//      • μ relaxes to a spatially uniform value,
//      • the final c profile equals the h-weighted phase mixture
//        h·cs(μ*) + (1−h)·cl(μ*) everywhere — NO interface solute excess,
//        the artefact that classical single-c (WBM/CH) models produce at
//        numerically widened interfaces.
// ---------------------------------------------------------------------------

#include "material/KKS.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/NoFluxBC.h"
#include "operators/Laplacian.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // Asymmetric, Fe-B-flavoured parameters: stiff solid parabola at low c,
    // soft liquid parabola at high c.
    const double ks = 4.0, cs0 = 0.2;
    const double kl = 1.0, cl0 = 0.7;
    KKSParabolic model(ks, cs0, kl, cl0);
    const KKSView v = model.view();

    // =======================================================================
    // 1. Partition identities
    // =======================================================================
    {
        const double cases[][2] = {
            {0.30, 0.0}, {0.30, 1.0}, {0.45, 0.5}, {0.62, 0.25},
            {0.21, 0.9}, {0.70, 0.1}, {0.05, 0.6},
        };
        for (const auto& tc : cases) {
            const Real c = static_cast<Real>(tc[0]);
            const Real h = static_cast<Real>(tc[1]);
            Real cs, cl, mu;
            v.partition(c, h, cs, cl, mu);

            require(std::fabs(ks * (cs - cs0) - mu) < 1e-12,
                    "solid chemical potential != mu");
            require(std::fabs(kl * (cl - cl0) - mu) < 1e-12,
                    "liquid chemical potential != mu");
            require(std::fabs(h * cs + (1.0 - h) * cl - c) < 1e-12,
                    "mixture rule violated");
        }
        // Single-phase limits
        require(std::fabs(v.cl(Real(0.55), Real(0)) - Real(0.55)) < 1e-14,
                "h=0 must give cl == c");
        require(std::fabs(v.cs(Real(0.25), Real(1)) - Real(0.25)) < 1e-14,
                "h=1 must give cs == c");
        // Interpolant endpoints and symmetry
        require(kks::h(Real(0)) == Real(0) && kks::h(Real(1)) == Real(1),
                "h(phi) endpoints wrong");
        require(std::fabs(kks::h(Real(0.5)) - Real(0.5)) < 1e-14,
                "h(0.5) != 0.5");
        require(kks::dh(Real(0)) == Real(0) && kks::dh(Real(1)) == Real(0),
                "dh endpoints wrong");
    }

    // =======================================================================
    // 2. Equilibrium and driving force
    // =======================================================================
    {
        const auto eq = model.equilibrium();
        require(std::fabs(eq.mu) < 1e-14, "bs==bl must give mu_eq == 0");
        require(std::fabs(eq.cs - cs0) < 1e-14 && std::fabs(eq.cl - cl0) < 1e-14,
                "equilibrium concentrations must be the parabola minima");

        // Equilibrium-consistent mixtures have zero driving force at any h
        for (double h = 0.0; h <= 1.0; h += 0.125) {
            const Real c = static_cast<Real>(h * eq.cs + (1.0 - h) * eq.cl);
            require(std::fabs(v.drivingForce(c, static_cast<Real>(h))) < 1e-13,
                    "driving force != 0 at equilibrium mixture");
        }
        // Supersaturated liquid must drive solidification (ΔG > 0)
        require(v.drivingForce(Real(0.5), Real(0)) > 0.0,
                "supersaturated liquid should favour solidification");
    }

    // =======================================================================
    // 3. GPU kernel vs CPU reference
    // =======================================================================
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 64, 1.0 / 64);
    {
        ScalarField c(mesh, "c", 1), h(mesh, "h", 1);
        ScalarField csG(mesh, "cs", 1), clG(mesh, "cl", 1), muG(mesh, "mu", 1);
        ScalarField csC(mesh, "csC", 1), clC(mesh, "clC", 1), muC(mesh, "muC", 1);
        c.initialize([](double x, double, double) { return 0.3 + 0.3 * x; });
        h.initialize([](double x, double, double) {
            return 0.5 * (1.0 + std::tanh((0.5 - x) / 0.05));
        });
        for (ScalarField* f : {&c, &h, &csG, &clG, &muG}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        kksPartitionOnGPU(model, c, h, csG, clG, muG);
        kksPartitionOnCPU(model, c, h, csC, clC, muC);
        csG.downloadCurrFromDevice();
        clG.downloadCurrFromDevice();
        muG.downloadCurrFromDevice();

        for (std::size_t i = 0; i < c.storedSize; ++i) {
            require(std::fabs(csG.curr[i] - csC.curr[i]) < 1e-14,
                    "GPU/CPU cs mismatch");
            require(std::fabs(clG.curr[i] - clC.curr[i]) < 1e-14,
                    "GPU/CPU cl mismatch");
            require(std::fabs(muG.curr[i] - muC.curr[i]) < 1e-14,
                    "GPU/CPU mu mismatch");
        }
    }

    // =======================================================================
    // 4. 1D relaxation to uniform chemical potential
    // =======================================================================
    {
        const int    N  = 64;
        const double dx = 1.0 / N;

        ScalarField c(mesh, "c", 1), h(mesh, "h", 1);
        ScalarField cs(mesh, "cs", 1), cl(mesh, "cl", 1), mu(mesh, "mu", 1);

        // Static solid-fraction profile (solid on the left)
        h.initialize([](double x, double, double) {
            return 0.5 * (1.0 - std::tanh((x - 0.5) / 0.05));
        });
        // Off-equilibrium start: solid at 0.25 (eq 0.2), liquid at 0.60 (eq 0.7)
        c.initialize([&h](double x, double, double) {
            const double hh = 0.5 * (1.0 - std::tanh((x - 0.5) / 0.05));
            return hh * 0.25 + (1.0 - hh) * 0.60;
        });
        for (ScalarField* f : {&c, &h, &cs, &cl, &mu}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        // dc/dt = ∇²μ  (M = 1), no-flux walls on both c and μ
        Equation eqC(c, "solute");
        eqC.setRHS(lap(mu, 1.0));
        NoFluxBC bcLoC(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC bcHiC(mesh.facePatch(Axis::X, Side::HIGH));
        NoFluxBC bcLoMu(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC bcHiMu(mesh.facePatch(Axis::X, Side::HIGH));

        const double dt = 0.05 * dx * dx;   // D_eff <= max(ks, kl) = 4
        Solver solver(eqC, {&bcLoC, &bcHiC}, dt, TimeScheme::EULER);

        const double sum0 = reduce::fieldSum(c);

        // Initial mu spread for the convergence assertion
        kksPartitionOnGPU(model, c, h, cs, cl, mu);
        const double spread0 = reduce::fieldMax(mu) - reduce::fieldMin(mu);

        const int nSteps = 120000;
        for (int s = 0; s < nSteps; ++s) {
            kksPartitionOnGPU(model, c, h, cs, cl, mu);
            bcLoMu.applyOnGPU(mu);
            bcHiMu.applyOnGPU(mu);
            solver.advance();
        }

        // (a) conservation
        const double sum1 = reduce::fieldSum(c);
        require(std::fabs(sum1 - sum0) < 1e-9 * std::fabs(sum0),
                "total solute not conserved: drift "
                + std::to_string(sum1 - sum0));

        // (b) mu uniform
        kksPartitionOnGPU(model, c, h, cs, cl, mu);
        const double muMax = reduce::fieldMax(mu);
        const double muMin = reduce::fieldMin(mu);
        const double spread1 = muMax - muMin;
        std::printf("  KKS relax: mu spread %.3e -> %.3e, mu* = %.6f\n",
                    spread0, spread1, 0.5 * (muMax + muMin));
        require(spread1 < 1e-3 * spread0 && spread1 < 1e-6,
                "chemical potential did not become uniform: spread "
                + std::to_string(spread1));

        // (c) NO interface excess: c must equal the h-weighted phase mixture
        //     of the FINAL uniform mu everywhere (incl. across the interface)
        const double muStar = 0.5 * (muMax + muMin);
        const double csStar = cs0 + muStar / ks;
        const double clStar = cl0 + muStar / kl;
        c.downloadCurrFromDevice();
        h.downloadCurrFromDevice();
        double maxDev = 0.0;
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(c.index(i));
            const double mix = h.curr[idx] * csStar
                             + (1.0 - h.curr[idx]) * clStar;
            maxDev = std::max(maxDev, std::fabs(c.curr[idx] - mix));
        }
        std::printf("  KKS relax: max |c - h*cs - (1-h)*cl| = %.3e\n", maxDev);
        require(maxDev < 1e-6,
                "interface solute excess detected: " + std::to_string(maxDev));
    }

    return 0;
}
