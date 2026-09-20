// ---------------------------------------------------------------------------
// float_smoke — core-path verification for PHIX_PRECISION=FLOAT builds.
//
// The strict module/convergence suites assume double tolerances and are not
// configured in FLOAT builds; this test covers the core numeric paths with
// float-appropriate tolerances:
//
//   1. Real is actually float (build wiring).
//   2. Device reductions vs CPU references (reductions accumulate in double
//      even for float fields).
//   3. 1D periodic diffusion vs analytic solution (full BC/operator/solver
//      loop).
//   4. Adaptive dt + NaN sentinel still function.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static_assert(sizeof(Real) == 4, "FLOAT build must have Real == float");

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // ---- 2. reductions ----------------------------------------------------
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        37, 0.1, 0.0, 23, 0.2, 0.0);
        ScalarField f(mesh, "f", 1);
        f.fillCurr(1e30);   // poison ghost (and physical, overwritten below)
        double refMax = -1e300, refSum = 0.0;
        for (int j = 0; j < 23; ++j)
        for (int i = 0; i < 37; ++i) {
            const Real v = static_cast<Real>(
                std::sin(0.37 * i + 0.61 * j) + 0.5);
            f.curr[static_cast<std::size_t>(f.index(i, j))] = v;
            refMax = std::max(refMax, static_cast<double>(v));
            refSum += static_cast<double>(v);
        }
        f.allocDevice();
        f.uploadAllToDevice();

        require(std::fabs(reduce::fieldMax(f) - refMax) < 1e-7,
                "fieldMax mismatch (float)");
        require(std::fabs(reduce::fieldSum(f) - refSum)
                    < 1e-6 * std::fabs(refSum) + 1e-6,
                "fieldSum mismatch (float)");
        require(!reduce::fieldHasNonFinite(f), "false NonFinite (float)");
    }

    // ---- 3. 1D periodic diffusion -----------------------------------------
    {
        const int    N  = 64;
        const double L  = 2.0 * M_PI;
        const double dx = L / N;
        const double dt = 0.2 * dx * dx;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

        ScalarField phi(mesh, "phi", 1);
        phi.initialize([](double x, double, double) { return std::sin(x); });
        phi.allocDevice();
        phi.uploadAllToDevice();

        Equation eq(phi, "diffusion");
        eq.setRHS(lap(phi, 1.0));
        PeriodicBC bc(mesh.facePatch(Axis::X, Side::LOW));
        Solver solver(eq, {&bc}, dt, TimeScheme::EULER);

        const int nSteps = 400;
        solver.run(nSteps);

        phi.downloadCurrFromDevice();
        const double decay = std::exp(-solver.time);
        double maxErr = 0.0;
        for (int i = 0; i < N; ++i)
            maxErr = std::max(maxErr, std::fabs(
                static_cast<double>(
                    phi.curr[static_cast<std::size_t>(phi.index(i))])
                - std::sin(mesh.coord(0, i)) * decay));
        // dx² spatial error ~1e-2·dx² plus float rounding; generous bound
        require(maxErr < 5e-3,
                "diffusion error too large (float): " + std::to_string(maxErr));
    }

    // ---- 4. adaptive dt + NaN sentinel ------------------------------------
    {
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
        ScalarField phi(mesh, "decay", 1);
        phi.fill(1.0);
        phi.allocDevice();
        phi.uploadAllToDevice();

        Equation eq(phi, "decay");
        eq.setRHS(pw(phi, PHIX_FN (Real p) { return -Real(10.0) * p; }));

        AdaptiveDt opts;
        opts.tol = 0.01; opts.dtMin = 1e-7; opts.dtMax = 0.05;
        opts.nanCheckEvery = 10;

        Solver solver(eq, {}, 1e-5, TimeScheme::EULER);
        solver.enableAdaptiveDt(opts);
        solver.run(200);

        require(solver.dt > 1e-5, "adaptive dt did not grow (float)");
        phi.downloadCurrFromDevice();
        const double v = phi.curr[static_cast<std::size_t>(phi.index(0))];
        require(v > 0.0 && v < 1.0 && std::isfinite(v),
                "decay result implausible (float)");
    }

    return 0;
}
