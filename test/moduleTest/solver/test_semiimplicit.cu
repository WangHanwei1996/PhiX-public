// ---------------------------------------------------------------------------
// module_semiimplicit — IMEX integrator (solver/SemiImplicitSolver.h)
//
// 1. Fully implicit diffusion at 100× the explicit stability limit,
//    verified against the EXACT discrete backward-Euler amplification of
//    the sin(x) eigenmode (machine-precision reference); the same dt makes
//    forward Euler blow up within 10 steps (demonstrated).
// 2. IMEX coupling: explicit reaction N = −λφ + implicit diffusion —
//    per-step factor (1 − λdt)/(1 + dt|λ_h|), again machine precision.
// 3. Backward-Euler order: error vs the continuous solution halves with
//    dt (measured p ≈ 1).
// 4. Cahn-Hilliard linear splitting: implicit −Mκ∇⁴ (BiharmonicOp) +
//    explicit M∇²f'(c), run at ~50× the explicit ∇⁴ limit: stable,
//    mass-conserving, and spinodal decomposition actually happens.
// ---------------------------------------------------------------------------

#include "solver/SemiImplicitSolver.h"
#include "solver/Solver.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
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
    const int    N  = 64;
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);
    PeriodicBC bc(mesh.facePatch(Axis::X, Side::LOW));

    // Discrete CD2 eigenvalue of the k=1 mode on the periodic grid
    const double lamH = (2.0 - 2.0 * std::cos(dx)) / (dx * dx);

    const double dtExp  = 0.5 * dx * dx;     // explicit stability limit (D=1)
    const double dtBig  = 100.0 * dtExp;

    // =======================================================================
    // 1. Fully implicit diffusion at 100× the explicit limit
    // =======================================================================
    {
        ScalarField phi(mesh, "phi", 1);
        phi.initialize([](double x, double, double) { return std::sin(x); });
        phi.allocDevice();
        phi.uploadAllToDevice();

        Equation eq(phi, "none");        // no RHS set → N ≡ 0
        LaplacianOp L(1.0, {&bc});
        SemiImplicitSolver::CGOptions cgo;
        cgo.relTol = 1e-12;
        SemiImplicitSolver semi(eq, {&bc}, L, dtBig, cgo);

        const int nSteps = 40;
        semi.run(nSteps);

        const double g = 1.0 / (1.0 + dtBig * lamH);   // BE amplification
        const double amp = std::pow(g, nSteps);
        phi.downloadCurrFromDevice();
        double err = 0.0;
        for (int i = 0; i < N; ++i)
            err = std::max(err, std::fabs(
                phi.curr[static_cast<std::size_t>(phi.index(i))]
                - std::sin(mesh.coord(0, i)) * amp));
        std::printf("  implicit diffusion @100x dt: err vs discrete-exact"
                    " %.2e (CG %d iters)\n", err, semi.lastSolve().iterations);
        require(err < 1e-8, "implicit diffusion mismatch: " + std::to_string(err));

        // Contrast: forward Euler at the same dt must explode
        ScalarField phiE(mesh, "phiE", 1);
        phiE.initialize([](double x, double, double) { return std::sin(x); });
        phiE.allocDevice();
        phiE.uploadAllToDevice();
        Equation eqE(phiE, "explicit");
        eqE.setRHS(lap(phiE, 1.0));
        Solver euler(eqE, {&bc}, dtBig, TimeScheme::EULER);
        euler.run(10);
        require(reduce::fieldMaxAbs(phiE) > 1e3,
                "forward Euler unexpectedly stable at 100x the limit");
    }

    // =======================================================================
    // 2. IMEX coupling — machine-precision discrete reference
    // =======================================================================
    {
        const double lam = 5.0;
        ScalarField phi(mesh, "phi", 1);
        phi.initialize([](double x, double, double) { return std::sin(x); });
        phi.allocDevice();
        phi.uploadAllToDevice();

        Equation eq(phi, "reaction");
        eq.setRHS(pw(phi, PHIX_FN (Real p) { return -Real(5.0) * p; }));
        LaplacianOp L(1.0, {&bc});
        SemiImplicitSolver::CGOptions cgo;
        cgo.relTol = 1e-12;
        const double dt = 10.0 * dtExp;          // λ·dt = 0.24 (FE-stable part)
        SemiImplicitSolver semi(eq, {&bc}, L, dt, cgo);

        const int nSteps = 60;
        semi.run(nSteps);

        const double g = (1.0 - lam * dt) / (1.0 + dt * lamH);
        const double amp = std::pow(g, nSteps);
        phi.downloadCurrFromDevice();
        double err = 0.0;
        for (int i = 0; i < N; ++i)
            err = std::max(err, std::fabs(
                phi.curr[static_cast<std::size_t>(phi.index(i))]
                - std::sin(mesh.coord(0, i)) * amp));
        std::printf("  IMEX reaction-diffusion: err vs discrete-exact %.2e\n",
                    err);
        require(err < 1e-8, "IMEX coupling mismatch: " + std::to_string(err));
    }

    // =======================================================================
    // 3. Backward-Euler temporal order ≈ 1 (vs continuous solution)
    // =======================================================================
    {
        auto errAt = [&](double dt) {
            ScalarField phi(mesh, "phi", 1);
            phi.initialize([](double x, double, double) { return std::sin(x); });
            phi.allocDevice();
            phi.uploadAllToDevice();
            Equation eq(phi, "none");
            LaplacianOp L(1.0, {&bc});
            SemiImplicitSolver::CGOptions cgo;
            cgo.relTol = 1e-13;
            SemiImplicitSolver semi(eq, {&bc}, L, dt, cgo);
            const int nSteps = static_cast<int>(std::lround(1.0 / dt));
            semi.run(nSteps);
            phi.downloadCurrFromDevice();
            // continuous decay of the k=1 mode uses the DISCRETE spatial
            // eigenvalue so only the temporal error is measured
            const double ref = std::exp(-lamH * semi.time);
            double e = 0.0;
            for (int i = 0; i < N; ++i)
                e = std::max(e, std::fabs(
                    phi.curr[static_cast<std::size_t>(phi.index(i))]
                    - std::sin(mesh.coord(0, i)) * ref));
            return e;
        };
        const double e1 = errAt(0.05), e2 = errAt(0.025), e3 = errAt(0.0125);
        const double p1 = std::log2(e1 / e2), p2 = std::log2(e2 / e3);
        std::printf("  BE temporal order: p = %.3f, %.3f\n", p1, p2);
        require(std::fabs(p1 - 1.0) < 0.15 && std::fabs(p2 - 1.0) < 0.15,
                "backward Euler order != 1");
    }

    // =======================================================================
    // 4. Cahn-Hilliard linear splitting at ~50× the explicit ∇⁴ limit
    // =======================================================================
    {
        const int    Nc  = 128;
        const double dxc = L0 / Nc;
        const double M   = 1.0, kappa = 2e-2;
        Mesh meshC = Mesh::makeUniform1D(CoordSys::CARTESIAN, Nc, dxc);
        PeriodicBC bcC(meshC.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcMu(meshC.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcLap(meshC.facePatch(Axis::X, Side::LOW));

        ScalarField c(meshC, "c", 1), muE(meshC, "muE", 1);
        c.initialize([](double x, double, double) {
            return 0.02 * (std::sin(3.0 * x) + std::sin(7.0 * x)
                           + std::cos(5.0 * x));
        });
        c.allocDevice();  c.uploadAllToDevice();
        muE.fill(0.0);
        muE.allocDevice(); muE.uploadAllToDevice();

        // explicit part: N(c) = M∇²(f'(c)), f'(c) = c³ − c (muE aux field)
        Equation eqMu(c, "mu");
        eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v * v * v - v; }));
        Equation eqC(c, "c");
        eqC.setRHS(lap(muE, M));

        BiharmonicOp Lc(M * kappa, {&bcC}, {&bcLap});
        // explicit ∇⁴ limit: dt ≈ 2/(Mκ·(π/dx)⁴)
        const double kmax  = M_PI / dxc;
        const double dtExp4 = 2.0 / (M * kappa * kmax * kmax * kmax * kmax);
        const double dt = 50.0 * dtExp4;
        SemiImplicitSolver::CGOptions cgo;
        cgo.relTol = 1e-10;
        cgo.maxIter = 2000;
        SemiImplicitSolver semi(eqC, {&bcC}, Lc, dt, cgo);

        const double sum0 = reduce::fieldSum(c);

        const int nSteps = static_cast<int>(std::lround(1.0 / dt));
        for (int s = 0; s < nSteps; ++s) {
            bcC.applyOnGPU(c);
            eqMu.computeRHS(muE);      // muE = f'(c)
            bcMu.applyOnGPU(muE);
            semi.advance();            // (I + dt·Mκ∇⁴) cⁿ⁺¹ = cⁿ + dt·M∇²muE
        }

        require(!reduce::fieldHasNonFinite(c), "CH splitting produced NaN/Inf");
        const double cMax  = reduce::fieldMaxAbs(c);
        const double drift = std::fabs(reduce::fieldSum(c) - sum0);
        std::printf("  CH @%.0fx dt: %d steps, max|c| = %.3f, mass drift %.2e,"
                    " last CG %d iters\n",
                    50.0, nSteps, cMax, drift, semi.lastSolve().iterations);
        require(cMax > 0.8 && cMax < 1.3,
                "CH spinodal decomposition did not develop as expected: max|c| = "
                + std::to_string(cMax));
        require(drift < 1e-9, "CH mass not conserved: "
                              + std::to_string(drift));
    }

    return 0;
}
