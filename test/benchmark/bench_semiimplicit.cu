// ---------------------------------------------------------------------------
// bench_semiimplicit — wall time per semi-implicit step.
//
//   implicit diff   2D fully implicit diffusion (LaplacianOp), dt at 100×
//                   the explicit limit — the Allen-Cahn-like configuration.
//   CH split        2D Cahn-Hilliard linear splitting (BiharmonicOp implicit,
//                   M∇²f'(c) explicit) at 50× the explicit ∇⁴ limit.
//
// Reports ms/step and the mean CG iteration count.  This is the tracking
// metric for the CG desynchronisation work: per-iteration host round trips
// dominate the step cost at small/medium grids.
// ---------------------------------------------------------------------------

#include "solver/SemiImplicitSolver.h"
#include "field/ScalarField.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>

using namespace PhiX;

static void benchImplicitDiffusion(int N) {
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));

    ScalarField phi(mesh, "phi", 1);
    phi.initialize([](double x, double y, double) {
        return std::sin(x) * std::cos(2.0 * y);
    });
    phi.allocDevice();
    phi.uploadAllToDevice();

    Equation eq(phi, "none");                    // N ≡ 0, fully implicit
    LaplacianOp L(1.0, {&bcx, &bcy});
    const double dt = 100.0 * 0.25 * dx * dx;    // 100× explicit 2D limit
    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol = 1e-8;
    SemiImplicitSolver semi(eq, {&bcx, &bcy}, L, dt, cgo);

    const int warmup = 5, iters = 100;
    long itSum = 0;
    for (int i = 0; i < warmup; ++i) semi.advance();
    cudaDeviceSynchronize();

    perf::WallTimer t;
    for (int i = 0; i < iters; ++i) {
        semi.advance();
        itSum += semi.lastSolve().iterations;
    }
    cudaDeviceSynchronize();
    const double ms = t.seconds() * 1e3 / iters;
    std::printf("  implicit diff N=%4d   %8.3f ms/step   CG %.1f iters/step\n",
                N, ms, static_cast<double>(itSum) / iters);
}

static void benchCahnHilliard(int N) {
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    const double M = 1.0, kappa = 2e-2;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));

    ScalarField c(mesh, "c", 1), muE(mesh, "muE", 1);
    c.initialize([](double x, double y, double) {
        return 0.05 * (std::sin(3.0 * x) * std::cos(2.0 * y)
                       + std::sin(5.0 * y) + std::cos(7.0 * x));
    });
    c.allocDevice();   c.uploadAllToDevice();
    muE.fill(0.0);
    muE.allocDevice(); muE.uploadAllToDevice();

    Equation eqMu(c, "mu");
    eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v * v * v - v; }));
    Equation eqC(c, "c");
    eqC.setRHS(lap(muE, M));

    BiharmonicOp Lc(M * kappa, {&bcx, &bcy}, {&bcx, &bcy});
    const double kmax = M_PI / dx;
    const double dt = 50.0 * 2.0 / (M * kappa * kmax * kmax * kmax * kmax);
    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol = 1e-8;
    cgo.maxIter = 2000;
    SemiImplicitSolver semi(eqC, {&bcx, &bcy}, Lc, dt, cgo);

    auto stepOnce = [&]() {
        bcx.applyOnGPU(c);
        bcy.applyOnGPU(c);
        eqMu.computeRHS(muE);
        bcx.applyOnGPU(muE);
        bcy.applyOnGPU(muE);
        semi.advance();
    };

    const int warmup = 5, iters = 100;
    long itSum = 0;
    for (int i = 0; i < warmup; ++i) stepOnce();
    cudaDeviceSynchronize();

    perf::WallTimer t;
    for (int i = 0; i < iters; ++i) {
        stepOnce();
        itSum += semi.lastSolve().iterations;
    }
    cudaDeviceSynchronize();
    const double ms = t.seconds() * 1e3 / iters;
    std::printf("  CH split      N=%4d   %8.3f ms/step   CG %.1f iters/step\n",
                N, ms, static_cast<double>(itSum) / iters);
}

int main() {
    std::printf("bench_semiimplicit (Real = %s, 2D)\n",
                sizeof(Real) == 8 ? "double" : "float");
    for (int N : {256, 512}) {
        benchImplicitDiffusion(N);
        benchCahnHilliard(N);
    }
    return 0;
}
