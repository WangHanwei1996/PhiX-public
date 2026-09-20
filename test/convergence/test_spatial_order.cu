// ---------------------------------------------------------------------------
// conv_spatial — measured convergence order of the spatial stencils.
//
// On sin(kx·x)·cos(ky·y) with analytically filled ghost cells, the L2 error
// of each operator is computed on a refinement sequence N = 32, 64, 128 and
// the observed order p = log2(err_N / err_2N) is asserted against nominal:
//
//   lap  CD2  → 2      lap  Iso9 → 2      lap  CD4 → 4
//   grad CD2  → 2      grad CD4  → 4
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "field/ScalarField.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static constexpr double KX = 1.0, KY = 2.0;

static double exact(double x, double y) {
    return std::sin(KX * x) * std::cos(KY * y);
}

// L2 error of `makeTerm(f)` against `ref(x,y)` on an N×N grid.
static double l2Error(int N, int ghost,
                      const std::function<Term(const ScalarField&)>& makeTerm,
                      const std::function<double(double, double)>& ref) {
    const double L  = 2.0 * M_PI;
    const double dx = L / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

    ScalarField f(mesh, "f", ghost);
    for (int j = -ghost; j < N + ghost; ++j)
    for (int i = -ghost; i < N + ghost; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            exact(mesh.coord(0, i), mesh.coord(1, j));
    f.allocDevice();
    f.uploadAllToDevice();

    Equation eq(f, "op");
    eq.setRHS(makeTerm(f));
    ScalarField rhs(mesh, "rhs", ghost);
    rhs.allocDevice();
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();

    double sumSq = 0.0;
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
        const double e = rhs.curr[static_cast<std::size_t>(rhs.index(i, j))]
                       - ref(mesh.coord(0, i), mesh.coord(1, j));
        sumSq += e * e;
    }
    return std::sqrt(sumSq / (static_cast<double>(N) * N));
}

// Measure orders on N = 32/64/128 and assert both against `nominal`.
static void checkOrder(const std::string& label, int ghost, double nominal,
                       const std::function<Term(const ScalarField&)>& makeTerm,
                       const std::function<double(double, double)>& ref) {
    const int Ns[3] = {32, 64, 128};
    double err[3];
    for (int m = 0; m < 3; ++m)
        err[m] = l2Error(Ns[m], ghost, makeTerm, ref);

    std::printf("  %-10s  err(32)=%.3e  err(64)=%.3e  err(128)=%.3e"
                "  p=%.3f, %.3f\n",
                label.c_str(), err[0], err[1], err[2],
                std::log2(err[0] / err[1]), std::log2(err[1] / err[2]));

    for (int m = 0; m < 2; ++m) {
        const double p = std::log2(err[m] / err[m + 1]);
        require(std::fabs(p - nominal) <= 0.2,
                label + ": measured order " + std::to_string(p)
                + " differs from nominal " + std::to_string(nominal));
    }
}

int main() {
    auto refLap = [](double x, double y) {
        return -(KX*KX + KY*KY) * exact(x, y);
    };
    auto refGx = [](double x, double y) {
        return KX * std::cos(KX*x) * std::cos(KY*y);
    };

    std::printf("spatial convergence orders:\n");

    checkOrder("lap CD2",  1, 2.0,
               [](const ScalarField& f) { return lap(f, "CD2", 1.0); }, refLap);
    checkOrder("lap Iso9", 1, 2.0,
               [](const ScalarField& f) { return lap(f, "Iso9", 1.0); }, refLap);
    checkOrder("lap CD4",  2, 4.0,
               [](const ScalarField& f) { return lap(f, "CD4", 1.0); }, refLap);
    checkOrder("lap CD6",  3, 6.0,
               [](const ScalarField& f) { return lap(f, "CD6", 1.0); }, refLap);
    checkOrder("grad CD2", 1, 2.0,
               [](const ScalarField& f) { return grad(f, 0, "CD2", 1.0); }, refGx);
    checkOrder("grad Iso9", 1, 2.0,
               [](const ScalarField& f) { return grad(f, 0, "Iso9", 1.0); }, refGx);
    checkOrder("grad CD4", 2, 4.0,
               [](const ScalarField& f) { return grad(f, 0, "CD4", 1.0); }, refGx);
    checkOrder("grad CD6", 3, 6.0,
               [](const ScalarField& f) { return grad(f, 0, "CD6", 1.0); }, refGx);

    return 0;
}
