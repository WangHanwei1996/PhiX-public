// ---------------------------------------------------------------------------
// module_adaptive_dt — rate-limited adaptive time stepping (AdaptiveDt.h)
//
// Uses linear decay problems (dphi/dt = -lambda*phi) where forward Euler is
// EXACTLY phi_{n+1} = phi_n * (1 - lambda*dt_n): recording each adapted dt on
// the host gives a machine-precision reference for the whole trajectory, so
// the test verifies both the controller maths and its plumbing into
// Solver / EquationSystem.
//
// Checks:
//   1. Solver (single-equation Euler): dt grows as the field decays, stays
//      inside [dtMin, dtMax], per-step change dt*max|RHS| <= tol, trajectory
//      matches the recorded-dt product, time == sum(dt).
//   2. EquationSystem (two coupled-slot decays, lambda 5 vs 50): stiffest
//      equation controls dt; both trajectories match their products.
//   3. NaN sentinel: an exploding RHS drives the field to Inf; with
//      nanCheckEvery=1 advance() must throw std::runtime_error.
//   4. enableAdaptiveDt on RK4 (Solver + EquationSystem) throws; bad options
//      throw std::invalid_argument.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "solver/Solver.h"
#include "field/ScalarField.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static void requireClose(double a, double b, double relTol,
                         const std::string& msg) {
    const double scale = std::max({std::fabs(a), std::fabs(b), 1e-300});
    if (std::fabs(a - b) > relTol * scale)
        throw std::runtime_error(msg + "  (" + std::to_string(a)
                                 + " vs " + std::to_string(b) + ")");
}

int main() {
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);

    // =======================================================================
    // 1. Solver: dphi/dt = -10*phi
    // =======================================================================
    {
        ScalarField phi(mesh, "phi", 1);
        phi.initialize([](double x, double, double) {
            return 1.0 + 0.5 * std::sin(x);
        });
        phi.allocDevice();
        phi.uploadAllToDevice();

        const double lam = 10.0;
        Equation eq(phi, "decay");
        eq.setRHS(pw(phi, PHIX_FN (double p) { return -10.0 * p; }));

        AdaptiveDt opts;
        opts.tol   = 0.01;
        opts.dtMin = 1e-9;
        opts.dtMax = 0.05;
        opts.grow  = 1.3;
        opts.safety = 0.9;
        opts.nanCheckEvery = 10;

        const double dt0 = 1e-6;
        Solver solver(eq, {}, dt0, TimeScheme::EULER);
        solver.enableAdaptiveDt(opts);

        double prod = 1.0, tsum = 0.0;
        bool grew = false;
        for (int s = 0; s < 300; ++s) {
            solver.advance();
            const double used = solver.dt;   // dt actually used this step
            require(used >= opts.dtMin && used <= opts.dtMax + 1e-15,
                    "Solver: dt left [dtMin, dtMax]");
            require(used * solver.adaptiveDt().lastMaxRate
                        <= opts.tol * (1.0 + 1e-12),
                    "Solver: per-step change exceeded tol");
            prod *= (1.0 - lam * used);
            tsum += used;
            if (used > dt0) grew = true;
        }
        require(grew, "Solver: dt never grew above its initial value");
        requireClose(solver.time, tsum, 1e-12, "Solver: time != sum(dt)");

        phi.downloadCurrFromDevice();
        for (int i = 0; i < mesh.n[0]; i += 5) {
            const double x   = mesh.coord(0, i);
            const double ref = (1.0 + 0.5 * std::sin(x)) * prod;
            requireClose(phi.curr[static_cast<std::size_t>(phi.index(i))],
                         ref, 1e-11, "Solver: trajectory mismatch");
        }
    }

    // =======================================================================
    // 2. EquationSystem: two decays, the stiffer one controls dt
    // =======================================================================
    {
        ScalarField a(mesh, "a", 1), b(mesh, "b", 1);
        a.initialize([](double, double, double) { return 1.0; });
        b.initialize([](double x, double, double) {
            return 1.0 + 0.5 * std::sin(x);
        });
        a.allocDevice(); a.uploadAllToDevice();
        b.allocDevice(); b.uploadAllToDevice();

        const double lamA = 5.0, lamB = 50.0;
        Equation eqA(a, "a");
        eqA.setRHS(pw(a, PHIX_FN (double p) { return -5.0 * p; }));
        Equation eqB(b, "b");
        eqB.setRHS(pw(b, PHIX_FN (double p) { return -50.0 * p; }));

        AdaptiveDt opts;
        opts.tol   = 0.02;
        opts.dtMin = 1e-9;
        opts.dtMax = 0.02;
        opts.grow  = 1.5;

        EquationSystem sys(1e-6, TimeScheme::EULER);
        sys.add(eqA, {});
        sys.add(eqB, {});
        sys.enableAdaptiveDt(opts);

        // Discrete max of b's initial profile (cell centres miss sin's peak)
        double maxB0 = 0.0;
        for (int i = 0; i < mesh.n[0]; ++i)
            maxB0 = std::max(maxB0, 1.0 + 0.5 * std::sin(mesh.coord(0, i)));

        double prodA = 1.0, prodB = 1.0;
        double maxRateSeen = 0.0;
        for (int s = 0; s < 200; ++s) {
            // Expected controlling rate BEFORE the step: max of both equations
            const double expRate = std::max(lamA * 1.0 * prodA,
                                            lamB * maxB0 * prodB);
            sys.advance();
            const double used = sys.dt;
            requireClose(sys.adaptiveDt().lastMaxRate, expRate, 1e-11,
                         "EquationSystem: controlling rate mismatch");
            require(used * sys.adaptiveDt().lastMaxRate
                        <= opts.tol * (1.0 + 1e-12),
                    "EquationSystem: per-step change exceeded tol");
            prodA *= (1.0 - lamA * used);
            prodB *= (1.0 - lamB * used);
            maxRateSeen = std::max(maxRateSeen, sys.adaptiveDt().lastMaxRate);
        }
        require(maxRateSeen > lamA * 1.0,
                "EquationSystem: stiff equation never controlled dt");

        a.downloadCurrFromDevice();
        b.downloadCurrFromDevice();
        requireClose(a.curr[static_cast<std::size_t>(a.index(3))],
                     prodA, 1e-11, "EquationSystem: field a mismatch");
        const double xb = mesh.coord(0, 7);
        requireClose(b.curr[static_cast<std::size_t>(b.index(7))],
                     (1.0 + 0.5 * std::sin(xb)) * prodB, 1e-11,
                     "EquationSystem: field b mismatch");
    }

    // =======================================================================
    // 3. NaN sentinel: exploding RHS must throw within a few steps
    // =======================================================================
    {
        ScalarField phi(mesh, "boom", 1);
        phi.initialize([](double, double, double) { return 1.0; });
        phi.allocDevice();
        phi.uploadAllToDevice();

        Equation eq(phi, "explode");
        eq.setRHS(pw(phi, PHIX_FN (double p) { return 1e150 * p; }));

        AdaptiveDt opts;
        opts.tol   = 0.01;
        opts.dtMin = 1e-8;      // rate clamp forces updates at dtMin
        opts.dtMax = 1.0;
        opts.nanCheckEvery = 1;

        Solver solver(eq, {}, 1e-8, TimeScheme::EULER);
        solver.enableAdaptiveDt(opts);

        bool threw = false;
        try {
            for (int s = 0; s < 10; ++s) solver.advance();
        } catch (const std::runtime_error&) { threw = true; }
        require(threw, "NaN sentinel did not fire on exploding field");
    }

    // =======================================================================
    // 4. Misuse must throw
    // =======================================================================
    {
        ScalarField phi(mesh, "p", 1);
        phi.allocDevice();
        Equation eq(phi, "e");
        eq.setRHS(pw(phi, PHIX_FN (double p) { return -p; }));

        AdaptiveDt opts;
        opts.tol = 0.01; opts.dtMin = 1e-9; opts.dtMax = 0.1;

        Solver rk4(eq, {}, 1e-3, TimeScheme::RK4);
        bool threw = false;
        try { rk4.enableAdaptiveDt(opts); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "RK4 Solver enableAdaptiveDt did not throw");

        EquationSystem sysRk4(1e-3, TimeScheme::RK4);
        threw = false;
        try { sysRk4.enableAdaptiveDt(opts); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "RK4 EquationSystem enableAdaptiveDt did not throw");

        Solver euler(eq, {}, 1e-3, TimeScheme::EULER);
        AdaptiveDt bad;   // tol == 0
        threw = false;
        try { euler.enableAdaptiveDt(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "invalid AdaptiveDt options did not throw");
    }

    return 0;
}
