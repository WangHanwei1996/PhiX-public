// ---------------------------------------------------------------------------
// bench_stencil — standard performance baseline for PhiX kernels.
//
// Reports, per grid size:
//   lap CD2 / lap CD4    raw Equation::computeRHS throughput (Mcells/s and
//                        approximate GB/s assuming ~3 double accesses/cell)
//   Euler step           full Solver::advance() incl. BCs, axpy, time-level
//                        memcpy and the per-step device synchronisation —
//                        the gap to the raw kernel row is framework overhead.
//
// Registered in ctest with modest sizes/repeats (< a few seconds); run the
// binary directly when profiling (combine with PHIX_ENABLE_NVTX=ON and
// PHIX_LINEINFO=ON for Nsight).
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
#include <string>

using namespace PhiX;

static void report(const std::string& label, int N, double msPerIter) {
    const double cells  = static_cast<double>(N) * N;
    const double mcps   = cells / (msPerIter * 1e3);            // Mcells/s
    const double gbps   = cells * 3.0 * sizeof(Real)
                          / (msPerIter * 1e6);                  // ~3 Reals/cell
    std::printf("  %-12s N=%5d   %8.3f ms/iter   %9.1f Mcells/s   ~%6.1f GB/s\n",
                label.c_str(), N, msPerIter, mcps, gbps);
}

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
    for (int i = 0; i < iters; ++i) {
        PHIX_NVTX_RANGE("bench/computeRHS");
        eq.computeRHS(rhs);
    }
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

    // advance() is asynchronous since v2.18.0 — bracket the loop with a
    // device sync so the wall time covers the actual GPU work.
    perf::WallTimer t;
    for (int i = 0; i < iters; ++i) {
        PHIX_NVTX_RANGE("bench/eulerStep");
        solver.advance();
    }
    cudaDeviceSynchronize();
    return t.seconds() * 1e3 / iters;
}

int main(int argc, char**) {
    const int warmup = 5;
    const int iters  = 50;

    std::printf("bench_stencil (Real = %s, 2D)\n",
                sizeof(Real) == 8 ? "double" : "float");
    for (int N : {512, 1024}) {
        report("lap CD2",    N, timeComputeRHS(N, "CD2", 1, warmup, iters));
        report("lap CD4",    N, timeComputeRHS(N, "CD4", 2, warmup, iters));
        report("euler step", N, timeEulerStep(N, warmup, iters));
    }
    (void)argc;
    return 0;
}
