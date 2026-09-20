// ---------------------------------------------------------------------------
// bench_mg_sweep — grid-independence (h-independence) of the MG-PCG iteration
// count, the defining evidence for a geometric multigrid preconditioner.
//
// Methodology follows test/moduleTest/solver/test_mg.cu: consistent system
// (b = A x_ref built through the operator itself) with a broadband per-cell
// hash reference field so conditioning actually bites.
//
// Two sweeps, both at FIXED PHYSICAL sigma (= a fixed implicit time step),
// refining the mesh — this is the regime a time-stepping solver actually
// meets: sigma/dx^2 grows as h -> 0, so plain CG degrades like 1/h while a
// multigrid preconditioner should stay flat.
//
//   S1  constant coefficient  a = 1
//   S2  variable coefficient  a(x,y) smooth log field, contrast 1e4
//
// Reports for each (N, sweep): MG-PCG iterations and wall time, plain-CG
// iterations and wall time, and the solution error of both against x_ref.
// ---------------------------------------------------------------------------

#include "solver/LinearSolver.h"
#include "solver/Preconditioner.h"
#include "solver/Multigrid.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "boundary/PeriodicBC.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
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
    const int N = f.mesh.n[0];
    for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < N; ++i) {
            double v = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1))
                     * 43758.5453;
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

static double maxErrVsRef(ScalarField& x, const ScalarField& xRef) {
    x.downloadCurrFromDevice();
    double err = 0.0;
    for (int j = 0; j < x.mesh.n[1]; ++j)
        for (int i = 0; i < x.mesh.n[0]; ++i) {
            const std::size_t idx = static_cast<std::size_t>(x.index(i, j, 0));
            err = std::max(err, std::fabs(
                static_cast<double>(x.curr[idx]) - xRef.curr[idx]));
        }
    return err;
}

struct Timing { int iters; double ms; double err; bool converged; };

static Timing timedSolve(ConjugateGradient& cg, LinearOperator& L, double sigma,
                         ScalarField& x, ScalarField& b, const ScalarField& xRef,
                         int maxIter, Preconditioner* M) {
    CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r  = cg.solve(L, sigma, x, b, 1e-10, maxIter, false, M);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return { r.iterations,
             std::chrono::duration<double, std::milli>(t1 - t0).count(),
             maxErrVsRef(x, xRef),
             r.converged };
}

int main() {
try {
    // Fixed physical sigma: the value that gives sigma*D/dx^2 = 500 on the
    // 256^2 reference mesh of module_mg.  Refining the mesh at this fixed
    // sigma raises sigma/dx^2 quadratically -- the honest stiff regime.
    const double dxRef  = 1.0 / 256.0;
    const double sigma  = 500.0 * dxRef * dxRef;
    const double CONTRAST = 1.0e4;

    std::printf("bench_mg_sweep (double, 2D, periodic)\n");
    std::printf("fixed physical sigma = %.6e  (= 500*dx^2 at N=256)\n", sigma);
    std::printf("relTol = 1e-10 on the true residual, both solvers\n\n");

    for (int sweep = 0; sweep < 2; ++sweep) {
        const bool varCoeff = (sweep == 1);
        std::printf("=== S%d  %s ===\n", sweep + 1,
                    varCoeff ? "variable coefficient, contrast 1e4"
                             : "constant coefficient a = 1");
        std::printf("%6s %10s | %8s %10s %10s | %8s %10s %10s | %8s\n",
                    "N", "sigma/dx^2", "MG it", "MG ms", "MG err",
                    "CG it", "CG ms", "CG err", "it ratio");

        for (int N : {128, 256, 512, 1024, 2048}) {
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
            ConjugateGradient cg(mesh, 1);

            MultigridPreconditioner mg(mesh,
                                       MultigridPreconditioner::BCKind::Periodic,
                                       MultigridPreconditioner::BCKind::Periodic);

            double *d_ax = nullptr, *d_ay = nullptr;
            LinearOperator* L = nullptr;
            LaplacianOp Lconst(1.0, bcs);
            VarCoeffLaplacianOp* Lvar = nullptr;

            if (!varCoeff) {
                mg.setup(1.0);
                L = &Lconst;
            } else {
                CUDA_CHECK(cudaMalloc(&d_ax, std::size_t(N + 1) * N * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_ay, std::size_t(N) * (N + 1) * sizeof(double)));
                std::vector<double> ax(std::size_t(N + 1) * N),
                                    ay(std::size_t(N) * (N + 1));
                auto aval = [&](double X, double Y) {
                    const double s = 0.5 * (1.0 + std::sin(2.0 * M_PI * X)
                                                * std::sin(2.0 * M_PI * Y));
                    return std::exp(std::log(CONTRAST) * s);
                };
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i <= N; ++i)
                        ax[std::size_t(i) + std::size_t(N + 1) * j]
                            = aval(i * dx, (j + 0.5) * dx);
                for (int j = 0; j <= N; ++j)
                    for (int i = 0; i < N; ++i)
                        ay[std::size_t(i) + std::size_t(N) * j]
                            = aval((i + 0.5) * dx, j * dx);
                CUDA_CHECK(cudaMemcpy(d_ax, ax.data(), ax.size() * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ay, ay.data(), ay.size() * sizeof(double),
                                      cudaMemcpyHostToDevice));
                mg.setupFaces(d_ax, d_ay);
                Lvar = new VarCoeffLaplacianOp(mesh, d_ax, d_ay, bcs);
                L = Lvar;
            }

            buildRhs(*L, sigma, xRef, b);

            const int mgCap = 2000;
            const int cgCap = 200000;
            Timing tmg = timedSolve(cg, *L, sigma, x, b, xRef, mgCap, &mg);
            Timing tcg = timedSolve(cg, *L, sigma, x, b, xRef, cgCap, nullptr);

            std::printf("%6d %10.1f | %8d %10.2f %10.2e | %8d %10.2f %10.2e | %8.1f%s\n",
                        N, sigma / (dx * dx),
                        tmg.iters, tmg.ms, tmg.err,
                        tcg.iters, tcg.ms, tcg.err,
                        tmg.iters > 0 ? double(tcg.iters) / tmg.iters : 0.0,
                        (tmg.converged && tcg.converged) ? "" : "  [NOT CONVERGED]");
            std::fflush(stdout);

            delete Lvar;
            if (d_ax) cudaFree(d_ax);
            if (d_ay) cudaFree(d_ay);
        }
        std::printf("\n");
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_mg_sweep FAILED: %s\n", e.what());
    return 1;
}
}
