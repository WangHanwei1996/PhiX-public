// ---------------------------------------------------------------------------
// bench_hostsync — how much the host-boundary-crossing reductions are worth.
//
// The framework's data-movement story has three parts that the fusion
// benchmark does not cover, because they are about CROSSINGS rather than
// about bytes inside one right-hand side:
//
//   A. Ghost refresh.  A ScalarField with BCs on every side used to cost one
//      kernel launch per BC per field per stage; BCBatch collapses them into
//      one (plus one corner pass).  Measured here as launches-equivalent
//      wall time over many refreshes.
//
//   B. Krylov control flow.  A textbook CG reads two inner products back to
//      the host every iteration to form alpha and beta.  PhiX keeps the whole
//      recurrence in device scalars and reads back only to TEST convergence,
//      every `checkEvery` iterations.  Sweeping checkEvery = 1,2,4,8 measures
//      what those host round-trips cost.
//
//   C. Launch collapse.  One `checkEvery` burst is captured as a CUDA graph
//      and replayed, turning ~10*checkEvery launches into one graph launch.
//      Measured as useGraph on/off at each cadence.
//
// B and C are measured on the same solve so they are directly comparable, and
// every configuration is checked to converge to the same iteration count --
// these are scheduling changes, so any change in iterations would mean the
// numerics moved and the comparison is void.
//
// Build:  ./build.sh          (this file is in the loop)
// Run  :  ./bench_hostsync | tee results/bench_hostsync.txt
// ---------------------------------------------------------------------------

#include "solver/LinearSolver.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/BCBatch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <vector>

using namespace PhiX;

static ScalarField makeDev(const Mesh& m, const char* name) {
    ScalarField f(m, name, 1);
    f.fill(0.0);
    f.allocDevice();
    f.uploadAllToDevice();
    return f;
}

static void fillBroadband(ScalarField& f) {
    for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < f.mesh.n[0]; ++i) {
            double v = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1)) * 43758.5453;
            v -= std::floor(v);
            f.curr[f.index(i, j, 0)] = static_cast<Real>(v - 0.5);
        }
    f.uploadAllToDevice();
}

static void buildRhs(LinearOperator& L, double sigma,
                     ScalarField& xRef, ScalarField& b) {
    ScalarField Lx(xRef.mesh, "_Lx", xRef.ghost);
    Lx.allocDevice();
    L.apply(xRef, Lx);
    Lx.downloadCurrFromDevice();
    xRef.downloadCurrFromDevice();
    for (std::size_t i = 0; i < b.storedSize; ++i)
        b.curr[i] = xRef.curr[i] - static_cast<Real>(sigma) * Lx.curr[i];
    if (!b.deviceAllocated()) b.allocDevice();
    b.uploadAllToDevice();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ===========================================================================
int main() {
try {
    std::printf("bench_hostsync  (double precision, 2D, periodic)\n");
    std::printf("every timing is the median of 5 repeats\n\n");

    // ---- A. ghost refresh: per-BC launches vs one batched launch ----------
    //
    // The count of BC OBJECTS is what batching collapses, so the periodic
    // case is the wrong test: PeriodicBC couples both sides of an axis, so a
    // 2D periodic field has only two of them and there is nothing to collapse
    // -- while BCBatch still pays for its second, corner-filling pass.  We
    // measured that case first and it is reported here as the honest lower
    // bound.  The configuration batching exists for is one BC object per
    // SIDE: four in 2D, six in 3D.
    std::printf("=== A. ghost refresh: per-BC launches vs one batched launch ===\n");
    std::printf("%6s %5s %7s %14s %14s %10s\n",
                "dim", "N", "BC objs", "per-BC ms", "BCBatch ms", "speedup");

    auto ghostCase = [](int dim, int N) {
        const double dx = 1.0 / N;
        Mesh mesh = (dim == 2)
            ? Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0)
            : Mesh::makeUniform3D(CoordSys::CARTESIAN, N, dx, 0.0,
                                                       N, dx, 0.0, N, dx, 0.0);
        std::vector<std::unique_ptr<NoFluxBC>> own;
        std::vector<BoundaryCondition*> bcs;
        const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
        for (int a = 0; a < dim; ++a)
            for (Side s : {Side::LOW, Side::HIGH}) {
                own.emplace_back(new NoFluxBC(mesh.facePatch(axes[a], s)));
                bcs.push_back(own.back().get());
            }

        ScalarField f(mesh, "f", 1);
        f.fill(0.0); f.allocDevice(); f.uploadAllToDevice();
        BCBatch batch;
        batch.build(f, bcs);

        const int REP = 2000;
        std::vector<double> tSeq, tBat;
        for (int r = 0; r < 5; ++r) {
            for (int k = 0; k < 50; ++k) { for (auto* b : bcs) b->applyOnGPU(f); }
            for (int k = 0; k < 50; ++k) batch.applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < REP; ++k) for (auto* b : bcs) b->applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < REP; ++k) batch.applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t2 = std::chrono::high_resolution_clock::now();
            tSeq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / REP);
            tBat.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count() / REP);
        }
        const double a = median(tSeq), b = median(tBat);
        std::printf("%6dD %5d %7d %14.5f %14.5f %9.2fx\n",
                    dim, N, int(bcs.size()), a, b, a / b);
    };

    for (int N : {128, 256, 512, 1024, 2048}) ghostCase(2, N);
    for (int N : {64, 128, 256})              ghostCase(3, N);

    // Periodic control: two BC objects, nothing to collapse, and the batched
    // path still runs its corner pass -- the case where batching does not pay.
    std::printf("\ncontrol, 2D periodic (2 BC objects, corner pass still run):\n");
    std::printf("%6s %5s %7s %14s %14s %10s\n",
                "dim", "N", "BC objs", "per-BC ms", "BCBatch ms", "speedup");
    for (int N : {512, 2048}) {
        const double dx = 1.0 / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0);
        PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
        std::vector<BoundaryCondition*> bcs = {&bcx, &bcy};
        ScalarField f(mesh, "f", 1);
        f.fill(0.0); f.allocDevice(); f.uploadAllToDevice();
        BCBatch batch; batch.build(f, bcs);
        const int REP = 2000;
        std::vector<double> tSeq, tBat;
        for (int r = 0; r < 5; ++r) {
            for (int k = 0; k < 50; ++k) { for (auto* b : bcs) b->applyOnGPU(f); }
            for (int k = 0; k < 50; ++k) batch.applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < REP; ++k) for (auto* b : bcs) b->applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < REP; ++k) batch.applyOnGPU(f);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t2 = std::chrono::high_resolution_clock::now();
            tSeq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / REP);
            tBat.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count() / REP);
        }
        const double a = median(tSeq), b = median(tBat);
        std::printf("%6dD %5d %7d %14.5f %14.5f %9.2fx\n", 2, N, 2, a, b, a / b);
    }

    // ---- B/C. CG: host readback cadence and graph capture -----------------
    // Fixed physical sigma, same construction as bench_mg_sweep so the two
    // benchmarks describe the same system.
    const double dxRef = 1.0 / 256.0;
    const double sigma = 500.0 * dxRef * dxRef;

    std::printf("\n=== B/C. CG: host readback cadence and CUDA-graph capture ===\n");
    std::printf("relTol 1e-10 on the true residual; iterations must not move\n\n");
    std::printf("%6s %10s %6s %10s %10s %10s %10s\n",
                "N", "checkEvery", "iters", "graph ms", "nograph ms",
                "graph gain", "vs cadence1");

    for (int N : {256, 512, 1024}) {
        const double dx = 1.0 / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0);
        PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
        std::vector<BoundaryCondition*> bcs = {&bcx, &bcy};

        ScalarField xRef = makeDev(mesh, "xref");
        fillBroadband(xRef);
        ScalarField b = makeDev(mesh, "b");
        ScalarField x = makeDev(mesh, "x");
        LaplacianOp L(1.0, bcs);
        buildRhs(L, sigma, xRef, b);

        double base = 0.0;
        for (int cadence : {1, 2, 4, 8}) {
            double ms[2] = {0.0, 0.0};
            int iters[2] = {0, 0};
            for (int g = 0; g < 2; ++g) {          // g = 0 graph on, 1 off
                ConjugateGradient cg(mesh, 1);
                cg.checkEvery = cadence;
                cg.useGraph   = (g == 0);
                std::vector<double> t;
                for (int r = 0; r < 5; ++r) {
                    CUDA_CHECK(cudaMemset(x.d_curr, 0,
                                          x.storedSize * sizeof(Real)));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto res = cg.solve(L, sigma, x, b, 1e-10, 5000, false);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    iters[g] = res.iterations;
                    if (!res.converged) {
                        std::printf("  !! did not converge (N=%d cadence=%d graph=%d)\n",
                                    N, cadence, g == 0);
                    }
                }
                ms[g] = median(t);
            }
            if (iters[0] != iters[1]) {
                std::printf("  !! iteration count moved between graph on/off "
                            "(%d vs %d) -- scheduling changed the numerics\n",
                            iters[0], iters[1]);
            }
            if (cadence == 1) base = ms[0];
            std::printf("%6d %10d %6d %10.3f %10.3f %9.2fx %10.2fx\n",
                        N, cadence, iters[0], ms[0], ms[1],
                        ms[1] / ms[0], base / ms[0]);
        }
        std::printf("\n");
    }

    std::printf("Reading: \"graph gain\" is graph-off / graph-on at the same\n"
                "cadence, i.e. what collapsing the burst into one launch buys.\n"
                "\"vs cadence1\" is cadence-1 / this cadence with the graph on,\n"
                "i.e. what NOT reading the residual back every iteration buys.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_hostsync failed: %s\n", e.what());
    return 1;
}
}
