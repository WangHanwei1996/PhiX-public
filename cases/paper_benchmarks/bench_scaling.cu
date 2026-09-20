// ---------------------------------------------------------------------------
// bench_scaling — problem-size sweep on a single GPU.
//
// PhiX is single-device by design, so a multi-GPU strong/weak scaling study is
// not available.  The honest substitute is a problem-size sweep that exposes
// where launch overhead stops dominating and the kernels become bandwidth
// saturated: fixed work per cell, grid refined from 256^2 to 4096^2.
//
// Reports per grid size:
//   lap CD2       raw Equation::computeRHS throughput (one stencil kernel)
//   euler step    full Solver::advance() — BCs, RHS, axpy, time-level
//                 bookkeeping; the gap to the raw row is framework overhead
//
// Structure and the ~3 Real accesses/cell traffic model follow
// test/benchmark/bench_stencil.cu so the two are directly comparable.
//
// Build:  ./build.sh          (see README.md in this directory)
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "field/ScalarField.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace PhiX;

static double timeComputeRHS(int N, const char* schemeName, int ghost,
                             int warmup, int iters) {
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);
    ScalarField f(mesh, "f", ghost);
    f.initialize([](double x, double y, double) {
        return std::sin(6.28 * x) * std::cos(6.28 * y);
    });
    f.allocDevice();
    f.uploadAllToDevice();

    Equation eq(f, "bench");
    eq.setRHS(lap(f, schemeName, 1.0));
    ScalarField rhs(mesh, "rhs", ghost);
    rhs.allocDevice();

    for (int i = 0; i < warmup; ++i) eq.computeRHS(rhs);
    cudaDeviceSynchronize();

    perf::CudaEventTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) eq.computeRHS(rhs);
    return t.stopMs() / iters;
}

static double timeEulerStep(int N, int warmup, int iters) {
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);
    ScalarField f(mesh, "f", 1);
    f.initialize([](double x, double y, double) {
        return std::sin(6.28 * x) * std::cos(6.28 * y);
    });
    f.allocDevice();
    f.uploadAllToDevice();

    Equation eq(f, "diffusion");
    eq.setRHS(lap(f, 1.0));
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
    Solver solver(eq, {&bcx, &bcy}, 0.1 * dx * dx, TimeScheme::EULER);

    for (int i = 0; i < warmup; ++i) solver.advance();
    cudaDeviceSynchronize();

    perf::WallTimer t;
    for (int i = 0; i < iters; ++i) solver.advance();
    cudaDeviceSynchronize();
    return t.seconds() * 1e3 / iters;
}

int main() {
try {
    const int warmup = 5;
    const int iters  = 50;

    std::printf("bench_scaling (Real = %s, 2D, single GPU)\n",
                sizeof(Real) == 8 ? "double" : "float");
    std::printf("%6s %12s | %10s %12s %10s | %10s %12s %10s | %8s\n",
                "N", "cells", "lap ms", "lap Mcell/s", "lap GB/s",
                "step ms", "step Mcell/s", "step GB/s", "overhead");

    for (int N : {256, 512, 1024, 2048, 4096}) {
        const double cells = static_cast<double>(N) * N;
        const double lapMs  = timeComputeRHS(N, "CD2", 1, warmup, iters);
        const double stepMs = timeEulerStep(N, warmup, iters);
        const double lapMc  = cells / (lapMs  * 1e3);
        const double stepMc = cells / (stepMs * 1e3);
        // Unavoidable minimum: each cell read once and written once (stencil
        // neighbours served from cache).  A strict lower bound on traffic;
        // write-allocate reads of the destination lines are NOT counted.
        // Was 3.0 (undocumented) until 2026-08-24 -- see paper FACTS.md 51.2.
        const double lapGB  = cells * 2.0 * sizeof(Real) / (lapMs  * 1e6);
        const double stepGB = cells * 2.0 * sizeof(Real) / (stepMs * 1e6);
        std::printf("%6d %12.3e | %10.4f %12.1f %10.1f | %10.4f %12.1f %10.1f | %7.2fx\n",
                    N, cells, lapMs, lapMc, lapGB, stepMs, stepMc, stepGB,
                    stepMs / lapMs);
        std::fflush(stdout);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_scaling FAILED: %s\n", e.what());
    return 1;
}
}
