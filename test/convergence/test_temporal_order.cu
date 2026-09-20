// ---------------------------------------------------------------------------
// conv_temporal — measured convergence order of the time integrators.
//
// dphi/dt = -phi (pointwise), phi(0) = 1, integrated to T = 1 with a dt
// halving sequence; error against exp(-T) gives the observed order:
//
//   EULER → 1        RK4 → 4
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "solver/Solver.h"
#include "field/ScalarField.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static double integrationError(TimeScheme scheme, double dt) {
    const double T = 1.0;
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 8, 0.1);

    ScalarField phi(mesh, "phi", 1);
    phi.fill(1.0);
    phi.allocDevice();
    phi.uploadAllToDevice();

    Equation eq(phi, "decay");
    eq.setRHS(pw(phi, PHIX_FN (double p) { return -p; }));

    Solver solver(eq, {}, dt, scheme);
    const int nSteps = static_cast<int>(std::lround(T / dt));
    solver.run(nSteps);

    phi.downloadCurrFromDevice();
    const double ref = std::exp(-solver.time);   // time = nSteps*dt exactly-ish
    return std::fabs(phi.curr[static_cast<std::size_t>(phi.index(0))] - ref);
}

static void checkOrder(const std::string& label, TimeScheme scheme,
                       double dt0, double nominal, double slack) {
    double err[3];
    for (int m = 0; m < 3; ++m)
        err[m] = integrationError(scheme, dt0 / (1 << m));

    std::printf("  %-6s  err(dt)=%.3e  err(dt/2)=%.3e  err(dt/4)=%.3e"
                "  p=%.3f, %.3f\n",
                label.c_str(), err[0], err[1], err[2],
                std::log2(err[0] / err[1]), std::log2(err[1] / err[2]));

    for (int m = 0; m < 2; ++m) {
        const double p = std::log2(err[m] / err[m + 1]);
        require(std::fabs(p - nominal) <= slack,
                label + ": measured order " + std::to_string(p)
                + " differs from nominal " + std::to_string(nominal));
    }
}

int main() {
    std::printf("temporal convergence orders:\n");
    checkOrder("EULER", TimeScheme::EULER, 0.1,  1.0, 0.15);
    checkOrder("RK4",   TimeScheme::RK4,   0.2,  4.0, 0.30);
    return 0;
}
