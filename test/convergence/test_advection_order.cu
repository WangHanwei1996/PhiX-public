// ---------------------------------------------------------------------------
// conv_advection — measured convergence order of the upwind advection schemes.
//
// u ≡ 1 (positive branch), f = sin(3x) with analytically filled ghosts;
// the operator error against 3·cos(3x) over N = 32/64/128 gives:
//
//   UW1 → 1        UW2 → 2        WENO5 → 5 (smooth field ⇒ optimal weights)
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "operators/Advection.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static double l2Error(int N, const std::string& scheme, int ghost) {
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

    ScalarField f(mesh, "f", ghost);
    for (int i = -ghost; i < N + ghost; ++i)
        f.curr[static_cast<std::size_t>(f.index(i))] =
            std::sin(3.0 * mesh.coord(0, i));
    f.allocDevice();
    f.uploadAllToDevice();

    VectorField u(mesh, "u", 1, ghost);
    for (int i = -ghost; i < N + ghost; ++i)
        u[0].curr[static_cast<std::size_t>(u[0].index(i))] = 1.0;
    u[0].allocDevice();
    u[0].uploadAllToDevice();

    Equation eq(f, "adv");
    eq.setRHS(adv(u, f, scheme, 1.0));
    ScalarField rhs(mesh, "rhs", ghost);
    rhs.allocDevice();
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();

    double sumSq = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = rhs.curr[static_cast<std::size_t>(rhs.index(i))]
                       - 3.0 * std::cos(3.0 * mesh.coord(0, i));
        sumSq += e * e;
    }
    return std::sqrt(sumSq / N);
}

static void checkOrder(const std::string& scheme, int ghost,
                       double nominal, double slack) {
    const int Ns[3] = {32, 64, 128};
    double err[3];
    for (int m = 0; m < 3; ++m)
        err[m] = l2Error(Ns[m], scheme, ghost);
    const double p1 = std::log2(err[0] / err[1]);
    const double p2 = std::log2(err[1] / err[2]);
    std::printf("  adv %-6s  err = %.3e / %.3e / %.3e   p = %.3f, %.3f\n",
                scheme.c_str(), err[0], err[1], err[2], p1, p2);
    require(std::fabs(p1 - nominal) <= slack
            && std::fabs(p2 - nominal) <= slack,
            scheme + ": measured order off nominal "
            + std::to_string(nominal));
}

int main() {
    std::printf("advection scheme convergence orders:\n");
    checkOrder("UW1",   1, 1.0, 0.15);
    checkOrder("UW2",   2, 2.0, 0.20);
    checkOrder("WENO5", 3, 5.0, 0.45);   // weights add mild pre-asymptotics
    return 0;
}
