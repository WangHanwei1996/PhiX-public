#include "equation/Equation.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static bool near(double a, double b) {
    return std::abs(a - b) < 1e-12;
}

int main() {
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    5, 1.0, 0.0,
                                    4, 1.0, 0.0);
    ScalarField phi(mesh, "phi", 1);
    for (int j = -1; j <= mesh.n[1]; ++j) {
        for (int i = -1; i <= mesh.n[0]; ++i) {
            phi.curr[phi.index(i, j)] = static_cast<double>(i * i + j);
        }
    }

    Term lap_t = lap(phi);
    Term grad_t = grad(phi, 0);
    require(lap_t.ghostRequired == 1, "lap ghost requirement mismatch");
    require(grad_t.ghostRequired == 1, "grad ghost requirement mismatch");

    Equation eq_lap(phi, "eq_lap");
    eq_lap.setRHS(lap_t);
    ScalarField rhs_lap(mesh, "rhs_lap", 1);
    eq_lap.computeRHSCPU(rhs_lap);
    require(eq_lap.requiredGhost() == 1, "Equation required ghost mismatch");
    require(near(rhs_lap.curr[rhs_lap.index(2, 2)], 2.0), "lap value mismatch");

    Equation eq_grad(phi, "eq_grad");
    eq_grad.setRHS(grad_t);
    ScalarField rhs_grad(mesh, "rhs_grad", 1);
    eq_grad.computeRHSCPU(rhs_grad);
    require(near(rhs_grad.curr[rhs_grad.index(2, 2)], 4.0), "grad value mismatch");

    return 0;
}
