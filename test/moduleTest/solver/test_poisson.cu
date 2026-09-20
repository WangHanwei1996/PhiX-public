// ---------------------------------------------------------------------------
// module_poisson — PoissonSolver (−D∇²Φ = rhs) with nullspace projection
//
// 1. Consistent system (periodic): b = −D∇²Φref via the operator; the
//    solver must recover Φref (both mean-zero) to CG tolerance.
// 2. Nullspace: a nonzero-mean rhs is projected; the solution is mean-zero
//    and satisfies −D∇²Φ = rhs − mean(rhs) (residual via operator apply).
// 3. No-flux variant (consistent system).
// 4. Analytic accuracy: rhs = D·(kx²+ky²-weighted modes) → Φ analytic,
//    CD2 discretisation error O(dx²).
// ---------------------------------------------------------------------------
#include "solver/LinearSolver.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"

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
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));

    // ---- 1. consistent system, periodic -------------------------------------
    {
        const double D = 1.7;
        LaplacianOp L(D, {&bcx, &bcy});
        ScalarField ref(mesh, "ref", 1), b(mesh, "b", 1), phi(mesh, "phi", 1);
        ref.initialize([](double x, double y, double) {
            return std::sin(2.0 * x) + 0.5 * std::cos(3.0 * y);
        });
        ref.allocDevice(); ref.uploadAllToDevice();
        b.allocDevice();
        L.apply(ref, b);                                  // b = D∇²ref
        // negate: rhs = −D∇²ref
        b.downloadCurrFromDevice();
        for (auto& v : b.curr) v = -v;
        b.uploadAllToDevice();

        phi.fill(0.0); phi.allocDevice(); phi.uploadAllToDevice();
        PoissonSolver poisson(mesh, 1, D, {&bcx, &bcy});
        auto res = poisson.solve(phi, b, 1e-12, 2000);
        std::printf("  periodic: %d iters, res %.2e\n",
                    res.iterations, res.relResidual);
        require(res.converged, "periodic Poisson did not converge");

        phi.downloadCurrFromDevice();
        ref.downloadCurrFromDevice();
        const double meanRef = reduce::fieldSum(ref) / (N * N);
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(phi.index(i, j));
            dev = std::max(dev, std::fabs(
                static_cast<double>(phi.curr[idx])
                - (ref.curr[idx] - meanRef)));
        }
        require(dev < 1e-9, "periodic Poisson solution off: "
                            + std::to_string(dev));
    }

    // ---- 2. nullspace projection --------------------------------------------
    {
        const double D = 1.0;
        ScalarField b(mesh, "b", 1), phi(mesh, "phi", 1), Ax(mesh, "Ax", 1);
        b.initialize([](double x, double y, double) {
            return std::sin(x) * std::cos(2.0 * y) + 7.5;   // mean ≈ 7.5 ≠ 0
        });
        b.allocDevice(); b.uploadAllToDevice();
        phi.fill(0.0); phi.allocDevice(); phi.uploadAllToDevice();
        Ax.allocDevice();

        PoissonSolver poisson(mesh, 1, D, {&bcx, &bcy});
        auto res = poisson.solve(phi, b, 1e-11, 2000);
        require(res.converged, "nonzero-mean rhs did not converge");
        require(std::fabs(reduce::fieldSum(phi)) < 1e-8 * N * N,
                "solution is not mean-zero");

        // residual: −D∇²phi vs (b − mean b)
        LaplacianOp L(D, {&bcx, &bcy});
        L.apply(phi, Ax);
        const double meanB = reduce::fieldSum(b) / (N * N);
        Ax.downloadCurrFromDevice();
        b.downloadCurrFromDevice();
        double rmax = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(b.index(i, j));
            rmax = std::max(rmax, std::fabs(
                -static_cast<double>(Ax.curr[idx])
                - (static_cast<double>(b.curr[idx]) - meanB)));
        }
        require(rmax < 1e-8, "equation residual too large: "
                             + std::to_string(rmax));
    }

    // ---- 3. no-flux consistent system ---------------------------------------
    {
        NoFluxBC nx0(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC nx1(mesh.facePatch(Axis::X, Side::HIGH));
        NoFluxBC ny0(mesh.facePatch(Axis::Y, Side::LOW));
        NoFluxBC ny1(mesh.facePatch(Axis::Y, Side::HIGH));
        std::vector<BoundaryCondition*> bcs = {&nx0, &nx1, &ny0, &ny1};

        const double D = 0.8;
        LaplacianOp L(D, bcs);
        ScalarField ref(mesh, "ref", 1), b(mesh, "b", 1), phi(mesh, "phi", 1);
        ref.initialize([&](double x, double y, double) {
            return std::cos(M_PI * x / L0) + 0.4 * std::cos(2.0 * M_PI * y / L0);
        });
        ref.allocDevice(); ref.uploadAllToDevice();
        b.allocDevice();
        L.apply(ref, b);
        b.downloadCurrFromDevice();
        for (auto& v : b.curr) v = -v;
        b.uploadAllToDevice();

        phi.fill(0.0); phi.allocDevice(); phi.uploadAllToDevice();
        PoissonSolver poisson(mesh, 1, D, bcs);
        auto res = poisson.solve(phi, b, 1e-12, 4000);
        std::printf("  no-flux:  %d iters, res %.2e\n",
                    res.iterations, res.relResidual);
        require(res.converged, "no-flux Poisson did not converge");

        phi.downloadCurrFromDevice();
        ref.downloadCurrFromDevice();
        const double meanRef = reduce::fieldSum(ref) / (N * N);
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(phi.index(i, j));
            dev = std::max(dev, std::fabs(
                static_cast<double>(phi.curr[idx])
                - (ref.curr[idx] - meanRef)));
        }
        require(dev < 1e-8, "no-flux Poisson solution off: "
                            + std::to_string(dev));
    }

    // ---- 4. analytic accuracy (O(dx²)) --------------------------------------
    {
        const double D = 1.0;
        ScalarField b(mesh, "b", 1), phi(mesh, "phi", 1);
        b.initialize([](double x, double y, double) {
            return 4.0 * std::sin(2.0 * x) + 9.0 * std::cos(3.0 * y);
        });
        b.allocDevice(); b.uploadAllToDevice();
        phi.fill(0.0); phi.allocDevice(); phi.uploadAllToDevice();

        PoissonSolver poisson(mesh, 1, D, {&bcx, &bcy});
        poisson.solve(phi, b, 1e-12, 2000);
        phi.downloadCurrFromDevice();
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double xa = mesh.coord(0, i), ya = mesh.coord(1, j);
            dev = std::max(dev, std::fabs(
                static_cast<double>(
                    phi.curr[static_cast<std::size_t>(phi.index(i, j))])
                - (std::sin(2.0 * xa) + std::cos(3.0 * ya))));
        }
        std::printf("  analytic: max dev %.2e (O(dx^2) ~ %.1e)\n",
                    dev, dx * dx);
        require(dev < 10.0 * dx * dx, "Poisson analytic error too large");
    }

    return 0;
}
