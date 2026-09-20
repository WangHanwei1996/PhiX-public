// ---------------------------------------------------------------------------
// conv_pde — end-to-end convergence of a full solve loop.
//
// 1D periodic diffusion  dphi/dt = ∇²phi,  phi(x,0) = sin(x)  on [0, 2π):
// exact solution phi(x,t) = sin(x)·exp(-t).  Integrated with RK4 and
// dt = 0.2·dx² (temporal error O(dt⁴) is negligible), so the measured
// order isolates the CD2 spatial error — but the whole pipeline runs:
// PeriodicBC ghost refresh, Equation::computeRHS, Solver time stepping.
//
// Nominal spatial order: 2.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "field/ScalarField.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static double diffusionError(int N) {
    const double L  = 2.0 * M_PI;
    const double dx = L / N;
    const double T  = 0.25;
    const double dt = 0.2 * dx * dx;

    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

    ScalarField phi(mesh, "phi", 1);
    phi.initialize([](double x, double, double) { return std::sin(x); });
    phi.allocDevice();
    phi.uploadAllToDevice();

    Equation eq(phi, "diffusion");
    eq.setRHS(lap(phi, 1.0));

    PeriodicBC bc(mesh.facePatch(Axis::X, Side::LOW));
    Solver solver(eq, {&bc}, dt, TimeScheme::RK4);

    const int nSteps = static_cast<int>(std::lround(T / dt));
    solver.run(nSteps);

    phi.downloadCurrFromDevice();
    const double decay = std::exp(-solver.time);
    double sumSq = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = phi.curr[static_cast<std::size_t>(phi.index(i))]
                       - std::sin(mesh.coord(0, i)) * decay;
        sumSq += e * e;
    }
    return std::sqrt(sumSq / N);
}

int main() {
    const int Ns[3] = {32, 64, 128};
    double err[3];
    for (int m = 0; m < 3; ++m)
        err[m] = diffusionError(Ns[m]);

    std::printf("1D periodic diffusion (CD2 + RK4, dt = 0.2 dx^2):\n"
                "  err(32)=%.3e  err(64)=%.3e  err(128)=%.3e  p=%.3f, %.3f\n",
                err[0], err[1], err[2],
                std::log2(err[0] / err[1]), std::log2(err[1] / err[2]));

    for (int m = 0; m < 2; ++m) {
        const double p = std::log2(err[m] / err[m + 1]);
        require(std::fabs(p - 2.0) <= 0.2,
                "PDE convergence order " + std::to_string(p) + " != 2");
    }
    return 0;
}
