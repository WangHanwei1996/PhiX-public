// ---------------------------------------------------------------------------
// module_mg — geometric multigrid preconditioner (solver/Multigrid.h) and
// the face-coefficient operator (VarCoeffLaplacianOp).
//
// Consistent-system methodology as module_linsolve/module_precond, with a
// BROADBAND (per-cell hash) reference so conditioning actually bites.
//
//   1. Operator equivalence: VarCoeffLaplacianOp with unit faces must match
//      LaplacianOp(D=1) to machine precision.
//   2. σ-flatness (the MG claim): constant-coefficient Helmholtz at
//      σD/dx² = 5, 500, 5e4 — MG-PCG iterations must stay ≤ 15 at every
//      stiffness (plain CG grows like √σ), solutions correct.  The σ sweep
//      re-solves through the same buffers, exercising graph replay across
//      per-solve σ changes.
//   3. Variable coefficients, contrast 1e4 (smooth log field): MG-PCG must
//      converge with < 1/3 of the plain-CG iterations, solution correct.
//   4. No-flux BCs, constant coefficient: converged + correct.
// ---------------------------------------------------------------------------

#include "solver/LinearSolver.h"
#include "solver/Preconditioner.h"
#include "solver/Multigrid.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static int nPass = 0;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
    ++nPass;
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

int main() {
try {
    const int    N  = 256;
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0,
                                    N, dx, 0.0);
    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bcx, &bcy};

    ScalarField xRef = makeDev(mesh, "xref");
    fillBroadband(xRef);
    ScalarField b = makeDev(mesh, "b");
    ScalarField x = makeDev(mesh, "x");
    ConjugateGradient cg(mesh, 1);

    // === 1. operator equivalence: unit faces == LaplacianOp ===============
    double *d_ax = nullptr, *d_ay = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ax, (N + 1) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ay, N * (N + 1) * sizeof(double)));
    {
        std::vector<double> ones((N + 1) * N, 1.0);
        CUDA_CHECK(cudaMemcpy(d_ax, ones.data(),
                              (N + 1) * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ay, ones.data(),
                              N * (N + 1) * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    LaplacianOp        Lconst(1.0, bcs);
    VarCoeffLaplacianOp Lvar(mesh, d_ax, d_ay, bcs);
    {
        ScalarField y1 = makeDev(mesh, "y1"), y2 = makeDev(mesh, "y2");
        Lconst.apply(xRef, y1);
        Lvar.apply(xRef, y2);
        y1.downloadCurrFromDevice();
        y2.downloadCurrFromDevice();
        double d = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const std::size_t id =
                    static_cast<std::size_t>(y1.index(i, j, 0));
                d = std::max(d, std::fabs(
                    static_cast<double>(y1.curr[id]) - y2.curr[id]));
            }
        require(d < 1e-10, "VarCoeffLaplacianOp(unit) != LaplacianOp, diff="
                            + std::to_string(d));
        std::printf("  [1] var-coeff(unit) == LaplacianOp (max diff %.2e)\n", d);
    }

    // === 2. σ-flatness: MG-PCG iterations stay ≤ 15 ======================
    MultigridPreconditioner mg(mesh,
                               MultigridPreconditioner::BCKind::Periodic,
                               MultigridPreconditioner::BCKind::Periodic);
    mg.setup(1.0);
    std::printf("  [2] hierarchy: %d levels\n", mg.levels());
    require(mg.levels() >= 6, "expected a deep hierarchy on 256^2");

    int cgIters500 = 0;
    for (double stiff : {5.0, 500.0, 5.0e4}) {
        const double sigma = stiff * dx * dx;
        buildRhs(Lconst, sigma, xRef, b);
        if (stiff == 500.0) {   // plain-CG reference at the middle stiffness
            CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
            auto rc = cg.solve(Lconst, sigma, x, b, 1e-10, 5000);
            cgIters500 = rc.iterations;
        }
        CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
        auto r = cg.solve(Lconst, sigma, x, b, 1e-10, 500, true, &mg);
        require(r.converged, "MG-PCG did not converge at stiffness "
                              + std::to_string(stiff));
        require(maxErrVsRef(x, xRef) < 1e-7,
                "MG-PCG wrong solution at stiffness " + std::to_string(stiff));
        require(r.iterations <= 15,
                "MG-PCG iterations not flat: " + std::to_string(r.iterations)
                + " at stiffness " + std::to_string(stiff));
        std::printf("      sigmaD/dx2=%-8g MG-PCG %d iters\n",
                    stiff, r.iterations);
    }
    std::printf("      (plain CG at 500: %d iters)\n", cgIters500);

    // === 3. variable coefficients, contrast 1e4 ===========================
    {
        std::vector<double> ax((N + 1) * N), ay(N * (N + 1));
        auto aval = [&](double X, double Y) {
            const double s = 0.5 * (1.0 + std::sin(2.0 * M_PI * X)
                                        * std::sin(2.0 * M_PI * Y));
            return std::exp(std::log(1.0e4) * s);   // 1 .. 1e4, smooth
        };
        for (int j = 0; j < N; ++j)
            for (int i = 0; i <= N; ++i)
                ax[i + (N + 1) * j] = aval(i * dx, (j + 0.5) * dx);
        for (int j = 0; j <= N; ++j)
            for (int i = 0; i < N; ++i)
                ay[i + N * j] = aval((i + 0.5) * dx, j * dx);
        CUDA_CHECK(cudaMemcpy(d_ax, ax.data(),
                              ax.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ay, ay.data(),
                              ay.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        mg.setupFaces(d_ax, d_ay);

        const double sigma = 500.0 * dx * dx;   // stiffness vs a_max ~ 5e6·dx²
        buildRhs(Lvar, sigma, xRef, b);
        CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
        auto rc = cg.solve(Lvar, sigma, x, b, 1e-10, 20000);
        require(rc.converged, "plain CG (var) did not converge");
        // contrast 1e4 => kappa(A) ~ 4e7: relTol 1e-10 bounds the solution
        // error only to ~kappa*tol ~ 4e-3 — use a conditioning-consistent
        // threshold (both solvers held to the same one).
        require(maxErrVsRef(x, xRef) < 2e-3, "plain CG (var) wrong solution");

        CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
        auto r = cg.solve(Lvar, sigma, x, b, 1e-10, 2000, true, &mg);
        require(r.converged, "MG-PCG (var) did not converge");
        require(maxErrVsRef(x, xRef) < 2e-3, "MG-PCG (var) wrong solution");
        require(r.iterations * 3 < rc.iterations,
                "MG-PCG (var, contrast 1e4) not ≥3× fewer iters: "
                + std::to_string(r.iterations) + " vs "
                + std::to_string(rc.iterations));
        std::printf("  [3] contrast 1e4: MG-PCG %d iters (CG %d)\n",
                    r.iterations, rc.iterations);
    }

    // === 4. no-flux BCs, constant coefficient =============================
    {
        NoFluxBC nbxL(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC nbxH(mesh.facePatch(Axis::X, Side::HIGH));
        NoFluxBC nbyL(mesh.facePatch(Axis::Y, Side::LOW));
        NoFluxBC nbyH(mesh.facePatch(Axis::Y, Side::HIGH));
        std::vector<BoundaryCondition*> nbcs = {&nbxL, &nbxH, &nbyL, &nbyH};
        LaplacianOp Ln(1.0, nbcs);
        MultigridPreconditioner mgN(mesh,
                                    MultigridPreconditioner::BCKind::NoFlux,
                                    MultigridPreconditioner::BCKind::NoFlux);
        mgN.setup(1.0);
        const double sigma = 500.0 * dx * dx;
        ScalarField xRefN = makeDev(mesh, "xrefn");
        fillBroadband(xRefN);
        ScalarField bN = makeDev(mesh, "bn");
        buildRhs(Ln, sigma, xRefN, bN);
        ScalarField xN = makeDev(mesh, "xn");
        ConjugateGradient cgN(mesh, 1);
        auto r = cgN.solve(Ln, sigma, xN, bN, 1e-10, 500, true, &mgN);
        require(r.converged, "no-flux MG-PCG did not converge");
        require(maxErrVsRef(xN, xRefN) < 1e-7,
                "no-flux MG-PCG wrong solution");
        require(r.iterations <= 15, "no-flux MG-PCG iterations not flat: "
                                     + std::to_string(r.iterations));
        std::printf("  [4] no-flux MG-PCG: %d iters\n", r.iterations);
    }

    cudaFree(d_ax);
    cudaFree(d_ay);
    std::printf("module_mg: %d assertions passed\n", nPass);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "module_mg FAILED: %s\n", e.what());
    return 1;
}
}
