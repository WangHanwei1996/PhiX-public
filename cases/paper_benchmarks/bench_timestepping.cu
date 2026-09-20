// ---------------------------------------------------------------------------
// bench_timestepping — what the semi-implicit path is worth on a production
// problem, at matched accuracy.
//
// The claim "the stiff term goes implicit, so the step can be 50x larger" is
// worthless on its own: a larger step that changes the answer is not a saving.
// This benchmark therefore never reports a speedup without the error that
// comes with it.
//
// Problem: Cahn-Hilliard spinodal decomposition on a periodic square,
//
//     dc/dt = M lap(mu),   mu = c^3 - c - kappa lap(c),
//
// integrated to a FIXED physical end time.  The fourth-order term is what
// makes it stiff: the explicit stability limit is dt < 2/(M kappa k_max^4),
// which shrinks like dx^4.
//
// Design:
//   Part A, at a size where an explicit reference is affordable.  Run explicit
//   Euler at its stability limit to t_end; that is the reference.  Then run
//   the semi-implicit split (biharmonic implicit through CG, chemical term
//   explicit) at 5x, 10x, 25x, 50x, 100x that step to the SAME t_end, and
//   report for each: wall clock, speedup, mass drift, whether the free energy
//   stayed monotone, and the error of the final field against the reference.
//   The last column is what makes the others meaningful.
//
//   Part B, at production sizes where explicit is no longer affordable.  Run
//   the semi-implicit path at the largest ratio Part A showed to be accurate,
//   and report the step count explicit integration WOULD have needed -- an
//   arithmetic consequence of the stability limit, not a measurement, and
//   labelled as such.
//
// Build:  ./build.sh          Run: ./bench_timestepping | tee results/...
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "equation/Equation.h"
#include "solver/SemiImplicitSolver.h"
#include "solver/LinearSolver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "core/CudaCheck.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace PhiX;

static const double M_MOB = 1.0;
static const double KAPPA = 2.0e-3;
static const double L0    = 2.0 * M_PI;

static Mesh makeMesh(int N) {
    const double dx = L0 / N;
    return Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);
}

// Deterministic multi-mode perturbation: identical for every run and every N,
// so the comparison is between integrators and not between initial states.
static void seedField(ScalarField& c) {
    c.initialize([](double x, double y, double) {
        double v = 0.0;
        for (int k = 1; k <= 6; ++k)
            v += std::sin(k * x + 0.7 * k * k) * std::cos((7 - k) * y + 0.3 * k);
        return 0.05 * v / 6.0;
    });
    c.allocDevice();
    c.uploadAllToDevice();
}

// Marginal linear stability of explicit Euler on the -M*kappa*grad^4 term:
// the amplification factor is exactly -1 at dt = 2/(M kappa k_max^4), so that
// value is oscillatory rather than stable and the nonlinear term grows on it.
// We verified this: at the marginal step the free energy RISES by 59 over the
// run, while every semi-implicit run is monotone.  The comparison therefore
// uses a safety factor, which is also what anyone integrating this explicitly
// would do; the ratios quoted later are against the USABLE explicit step, not
// against a step that does not work.
static const double SAFETY = 0.4;

static double explicitMarginal(int N) {
    const double dx   = L0 / N;
    const double kmax = M_PI / dx;
    return 2.0 / (M_MOB * KAPPA * kmax * kmax * kmax * kmax);
}
static double explicitLimit(int N) { return SAFETY * explicitMarginal(N); }

// F = integral [ (c^2-1)^2/4 + kappa/2 |grad c|^2 ]
//
// fieldGradSq differentiates with CD2 and therefore READS THE GHOST CELLS, so
// they must be refreshed first.  Sampling straight after a time step leaves
// them stale, which is the bug this signature exists to prevent.
static double freeEnergy(ScalarField& c, std::vector<BoundaryCondition*>& bcs) {
    for (auto* b : bcs) b->applyOnGPU(c);
    const double dV = c.mesh.d[0] * c.mesh.d[1];
    const double bulk = reduce::fieldSumPW(c, PHIX_FN (Real v) {
        const Real t = v * v - Real(1);
        return t * t * Real(0.25);
    });
    return dV * (bulk + 0.5 * KAPPA * reduce::fieldGradSq(c));
}

struct Run {
    double ms;          // wall clock
    double massDrift;
    double energyRise;  // largest increase between samples (should be <= 0)
    int    cgIters;     // last solve
};

// ---- explicit Euler on the full right-hand side ---------------------------
static Run runExplicit(ScalarField& c, std::vector<BoundaryCondition*>& bcs,
                       double dt, long nSteps, int samples, bool trace = false) {
    ScalarField muE(c.mesh, "muE", 1);
    muE.fill(0.0); muE.allocDevice(); muE.uploadAllToDevice();

    // The explicit reference must solve the SAME equation as the semi-implicit
    // path: mu = f'(c) - kappa lap(c).  Dropping the gradient term leaves
    // dc/dt = M lap(c^3 - c), which is backward diffusion in the spinodal
    // region -- it produces cell-scale oscillation, a huge |grad c| and a
    // rising free energy while max|c| stays near 1.  That is what the energy
    // guard caught the first time this benchmark was run.
    Equation eqMu(c, "mu");
    eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v * v * v - v; })
              + lap(c, -KAPPA));
    Equation eqC(c, "c");
    eqC.setRHS(lap(muE, M_MOB));

    const double mass0 = reduce::fieldSum(c);
    double  Fprev = freeEnergy(c, bcs), rise = 0.0;
    const long every = std::max<long>(1, nSteps / samples);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long s = 1; s <= nSteps; ++s) {
        for (auto* b : bcs) b->applyOnGPU(c);
        eqMu.computeRHS(muE);
        for (auto* b : bcs) b->applyOnGPU(muE);
        eqC.advanceTransient({}, dt, &c);
        if (s % every == 0) {
            const double F = freeEnergy(c, bcs);
            if (trace)
                std::printf("    trace step %8ld  F=%12.6f  max|c|=%9.4f\n",
                            s, F, reduce::fieldMaxAbs(c));
            rise = std::max(rise, F - Fprev);
            Fprev = F;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return { std::chrono::duration<double, std::milli>(t1 - t0).count(),
             reduce::fieldSum(c) - mass0, rise, 0 };
}

// ---- semi-implicit: biharmonic implicit, chemical term explicit -----------
static Run runSemi(ScalarField& c, std::vector<BoundaryCondition*>& bcs,
                   double dt, long nSteps, int samples) {
    ScalarField muE(c.mesh, "muE", 1);
    muE.fill(0.0); muE.allocDevice(); muE.uploadAllToDevice();

    Equation eqMu(c, "mu");
    eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v * v * v - v; }));
    Equation eqC(c, "c");
    eqC.setRHS(lap(muE, M_MOB));
    BiharmonicOp L(M_MOB * KAPPA, bcs, bcs);

    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol  = 1e-8;
    cgo.maxIter = 2000;
    SemiImplicitSolver semi(eqC, bcs, L, dt, cgo);

    const double mass0 = reduce::fieldSum(c);
    double  Fprev = freeEnergy(c, bcs), rise = 0.0;
    const long every = std::max<long>(1, nSteps / samples);
    int lastIters = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long s = 1; s <= nSteps; ++s) {
        for (auto* b : bcs) b->applyOnGPU(c);
        eqMu.computeRHS(muE);
        for (auto* b : bcs) b->applyOnGPU(muE);
        semi.advance();
        lastIters = semi.lastSolve().iterations;
        if (s % every == 0) {
            const double F = freeEnergy(c, bcs);
            rise = std::max(rise, F - Fprev);
            Fprev = F;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return { std::chrono::duration<double, std::milli>(t1 - t0).count(),
             reduce::fieldSum(c) - mass0, rise, lastIters };
}

static void errVs(ScalarField& a, ScalarField& b, double& linf, double& l2) {
    a.downloadCurrFromDevice();
    b.downloadCurrFromDevice();
    linf = 0.0; double s2 = 0.0; long n = 0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
        for (int i = 0; i < a.mesh.n[0]; ++i) {
            const std::size_t k = std::size_t(a.index(i, j, 0));
            const double d = double(a.curr[k]) - double(b.curr[k]);
            linf = std::max(linf, std::fabs(d));
            s2 += d * d; ++n;
        }
    l2 = std::sqrt(s2 / n);
}

int main() {
try {
    std::printf("bench_timestepping — Cahn-Hilliard spinodal, "
                "explicit vs semi-implicit at matched end time\n");
    std::printf("M = %.1f, kappa = %.1e, domain (2 pi)^2, periodic\n\n",
                M_MOB, KAPPA);

    // ================= Part A: accuracy-controlled comparison ==============
    const int    NA    = 512;
    const double dtExp = explicitLimit(NA);
    const double tEnd  = 0.20;   // far enough for the pattern to saturate;
                             // at 0.10 max|c| is still ~0.5 and the
                             // accuracy comparison would be between
                             // three nearly-flat fields
    const long   nExp  = std::lround(tEnd / dtExp);

    std::printf("=== A. %d^2, t_end = %.2f ===\n", NA, tEnd);
    std::printf("marginal explicit step  %.4e  (amplification factor -1; unusable)\n",
                explicitMarginal(NA));
    std::printf("usable explicit step    %.4e  = %.2f x marginal -> %ld steps\n\n",
                dtExp, SAFETY, nExp);

    Mesh meshA = makeMesh(NA);
    PeriodicBC bxA(meshA.facePatch(Axis::X, Side::LOW));
    PeriodicBC byA(meshA.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcsA = {&bxA, &byA};

    ScalarField ref(meshA, "ref", 1);
    seedField(ref);
    Run r0 = runExplicit(ref, bcsA, dtExp, nExp, 20, true);
    if (r0.energyRise > 1e-6) {
        std::printf("\nABORT: the explicit reference is not stable at this step "
                    "(free energy rose by %.2e).\nLower SAFETY and re-run; a "
                    "reference that drifts is not a reference.\n", r0.energyRise);
        return 1;
    }
    std::printf("%-14s %10s %9s %11s %11s %11s %9s\n",
                "integrator", "wall (s)", "speedup", "mass drift",
                "max dF", "Linf vs ref", "CG its");
    std::printf("%-14s %10.2f %9s %11.2e %11.2e %11s %9s\n",
                "explicit", r0.ms / 1e3, "1.00x", r0.massDrift,
                r0.energyRise, "(reference)", "-");

    // 5x was measured and is 0.55x -- SLOWER than explicit, because each step
// carries a CG solve.  The crossover is above it, so the sweep starts at 10x.
    for (double ratio : {10.0, 25.0, 50.0, 100.0, 200.0}) {
        const double dt = ratio * dtExp;
        const long   n  = std::lround(tEnd / dt);
        ScalarField c(meshA, "c", 1);
        seedField(c);
        Run r = runSemi(c, bcsA, dt, n, 20);
        double linf, l2;
        errVs(c, ref, linf, l2);
        char name[32];
        std::snprintf(name, sizeof name, "semi %gx", ratio);
        std::printf("%-14s %10.2f %8.2fx %11.2e %11.2e %11.2e %9d\n",
                    name, r.ms / 1e3, r0.ms / r.ms, r.massDrift,
                    r.energyRise, linf, r.cgIters);
    }

    // ================= Part B: production sizes ============================
    //
    // The step is held FIXED in physical time -- the value Part A showed to be
    // accurate at 512^2 -- rather than pegged to a multiple of each mesh's
    // explicit limit.  That is what a user actually does: the step is chosen
    // for accuracy, and the explicit limit is whatever it happens to be.  It
    // also makes the point the other design would have hidden: the explicit
    // limit collapses as dx^4, so the ratio grows by 16 for every halving of
    // the mesh while the semi-implicit run costs the same number of steps.
    const double dtProd = 50.0 * dtExp;          // accurate at 512^2, see A
    const long   nProd  = std::lround(tEnd / dtProd);

    std::printf("\n=== B. production sizes, fixed dt = %.3e (%ld steps) ===\n",
                dtProd, nProd);
    std::printf("%6s %14s %12s %12s %12s %11s\n",
                "N", "dt/dt_explicit", "ms/step", "wall (s)",
                "mass drift", "CG its");
    for (int N : {512, 1024, 2048}) {
        Mesh mesh = makeMesh(N);
        PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
        std::vector<BoundaryCondition*> bcs = {&bx, &by};
        ScalarField c(mesh, "c", 1);
        seedField(c);
        Run r = runSemi(c, bcs, dtProd, nProd, 5);
        std::printf("%6d %13.0fx %12.4f %12.2f %12.2e %11d\n",
                    N, dtProd / explicitLimit(N), r.ms / nProd, r.ms / 1e3,
                    r.massDrift, r.cgIters);
    }
    std::printf("\nThe ratio column is arithmetic from the stability limit,\n"
                "not a timing: it is what explicit integration would have\n"
                "been forced to at that mesh.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_timestepping failed: %s\n", e.what());
    return 1;
}
}
