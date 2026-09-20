// ---------------------------------------------------------------------------
// rotating_scalar — scalar transport in a solid-body rotation field.
//
// A second deliberately NON-phase-field case, and the one that exercises the
// high-order advection line end-to-end.  test/convergence/conv_advection
// measures the *operator* error of UW1 / UW2 / WENO5 against an analytical
// derivative; this case measures what a user actually cares about: the error
// after a full revolution of time marching, with boundary conditions and a
// time integrator in the loop.
//
//   df/dt + u . grad f = 0 ,    u = omega ( -(y-yc), (x-xc) ) ,  omega = 2 pi
//
// One revolution takes t = 1.  The velocity field is divergence free, so
// u . grad f = div(u f) and the non-conservative `adv` operator transports
// exactly.  The exact solution is the initial field rigidly rotated, so after
// an integer number of revolutions it equals the initial condition — the error
// is then purely numerical dissipation and dispersion.
//
// Two initial fields:
//   gauss   smooth Gaussian bump      -> order of accuracy is meaningful
//   slot    Zalesak slotted cylinder  -> shape preservation / over-under-shoot
//
// Usage:  ./rotating_scalar [ic] [Nbase] [nRev]
//           ic     gauss | slot      (default gauss)
//           Nbase  coarsest grid     (default 64; refines Nbase,2N,4N)
//           nRev   revolutions       (default 1)
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Advection.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace PhiX;

namespace {

constexpr double OMEGA = 2.0 * M_PI;      // one revolution per unit time
constexpr double XC = 0.5, YC = 0.5;
constexpr double R0 = 0.15, X0 = 0.5, Y0 = 0.25;   // blob centre, offset from XC

double icGauss(double x, double y) {
    const double r2 = (x - X0) * (x - X0) + (y - Y0) * (y - Y0);
    return std::exp(-r2 / (2.0 * (0.06 * 0.06)));
}

double icSlot(double x, double y) {
    const double r = std::sqrt((x - X0) * (x - X0) + (y - Y0) * (y - Y0));
    if (r > R0) return 0.0;
    // slot: width 0.05, reaching up to y = Y0 + 0.06
    if (std::fabs(x - X0) < 0.025 && y < Y0 + 0.06) return 0.0;
    return 1.0;
}

struct Res { double l2, linf, fmin, fmax; };

Res runOnce(int N, const std::string& ic, const std::string& scheme,
            double nRev, double* msPerStep) {
    const double dx = 1.0 / N;
    const int ghost = (scheme == "WENO5") ? 3 : ((scheme == "UW2") ? 2 : 1);
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

    auto icf = (ic == "slot") ? icSlot : icGauss;

    ScalarField f(mesh, "f", ghost);
    f.initialize([&](double x, double y, double) { return icf(x, y); });
    f.allocDevice();
    f.uploadAllToDevice();

    VectorField u(mesh, "u", 2, ghost);
    u.initializeComponent(0, [](double, double y, double) { return -OMEGA * (y - YC); });
    u.initializeComponent(1, [](double x, double,   double) { return  OMEGA * (x - XC); });
    u.allocDevice();
    u.uploadAllToDevice();

    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bx, &by};

    // df/dt = -(u . grad f)
    Equation eq(f, "rot");
    eq.setRHS(adv(u, f, scheme, -1.0));

    // CFL on the maximum rotational speed at the domain corner.
    const double umax = OMEGA * std::sqrt(0.5);
    const double dt   = 0.2 * dx / umax;
    const int nSteps  = static_cast<int>(nRev / dt + 0.5);

    Solver solver(eq, bcs, nRev / nSteps, TimeScheme::RK4);

    cudaDeviceSynchronize();
    perf::WallTimer wall;
    for (int s = 0; s < nSteps; ++s) solver.advance();
    cudaDeviceSynchronize();
    *msPerStep = wall.seconds() * 1e3 / nSteps;

    f.downloadCurrFromDevice();
    double e2 = 0.0, einf = 0.0, fmin = 1e300, fmax = -1e300;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j);
            const double ex = icf(x, y);                 // integer revolutions
            const double d  = static_cast<double>(f.curr[f.index(i, j, 0)]) - ex;
            e2 += d * d;
            einf = std::max(einf, std::fabs(d));
            fmin = std::min(fmin, static_cast<double>(f.curr[f.index(i, j, 0)]));
            fmax = std::max(fmax, static_cast<double>(f.curr[f.index(i, j, 0)]));
        }
    return { std::sqrt(e2 / (double(N) * N)), einf, fmin, fmax };
}

}  // namespace

int main(int argc, char** argv) {
try {
    const std::string ic = (argc > 1) ? argv[1] : "gauss";
    const int    Nb   = (argc > 2) ? std::atoi(argv[2]) : 64;
    const double nRev = (argc > 3) ? std::atof(argv[3]) : 1.0;

    std::printf("rotating_scalar — solid-body rotation, ic=%s, %.0f revolution(s)\n",
                ic.c_str(), nRev);
    std::printf("  exact solution after an integer number of revolutions is the "
                "initial field\n\n");

    for (const std::string scheme : {"UW1", "UW2", "WENO5"}) {
        std::printf("--- %s ---\n", scheme.c_str());
        std::printf("%6s %14s %10s %14s %12s %12s %10s\n",
                    "N", "L2 err", "order", "Linf err", "min", "max", "ms/step");
        double prev = 0.0;
        for (int i = 0; i < 3; ++i) {
            const int N = Nb << i;
            double ms = 0.0;
            Res r = runOnce(N, ic, scheme, nRev, &ms);
            if (i == 0)
                std::printf("%6d %14.6e %10s %14.6e %12.5f %12.5f %10.4f\n",
                            N, r.l2, "-", r.linf, r.fmin, r.fmax, ms);
            else
                std::printf("%6d %14.6e %10.3f %14.6e %12.5f %12.5f %10.4f\n",
                            N, r.l2, std::log(prev / r.l2) / std::log(2.0),
                            r.linf, r.fmin, r.fmax, ms);
            prev = r.l2;
            std::fflush(stdout);
        }
        std::printf("\n");
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rotating_scalar FAILED: %s\n", e.what());
    return 1;
}
}
