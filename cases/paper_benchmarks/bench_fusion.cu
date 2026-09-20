// ---------------------------------------------------------------------------
// bench_fusion — ablation: EvalPlan (per-step launchers) vs FusedTerm
// (compile-time fused kernel) on the real MPF_AC_DW mu_0 expression
// (10 terms: 2 pointwise double-well + 8 gradient/Laplacian cross terms).
//
// Paths:
//   A  ExprTree -> EvalPlan     one mu field, sequential launcher steps
//   B  FusedTerm single         one mu field, single fused kernel
//   A3 ExprTree x3              three mu fields (real app workload)
//   C  fuse_multi_compute       three mu fields, ONE kernel launch
//
// Reports ms/eval, Mcells/s, EvalPlan step count, and max|A-B| (round-off).
// ---------------------------------------------------------------------------
#include "equation/Equation.h"
#include "equation/EvalPlan.h"
#include "equation/Expr.h"
#include "equation/Term.h"
#include "equation/TermPW.inl"
#include "equation/FieldOps.inl"
#include "equation/FusedTerm.h"
#include "boundary/PeriodicBC.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>

using namespace PhiX;

static double timeIt(const std::function<void()>& fn, int warm, int iters) {
    for (int i = 0; i < warm; ++i) fn();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    const double W = 1.0, e2a = 0.4, e2b = 0.4, e2ab = 0.4;

    for (int N : {512, 1024, 2048}) {
        const double dx = 1.0 / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

        ScalarField phi0(mesh, "phi0", 1), phia(mesh, "phia", 1), phib(mesh, "phib", 1);
        phi0.initialize([](double x, double y, double) {
            return 0.5 + 0.4 * std::sin(6.28318 * x) * std::cos(6.28318 * y); });
        phia.initialize([](double x, double y, double) {
            return 0.25 + 0.2 * std::cos(12.566 * x) * std::sin(6.28318 * y); });
        phib.initialize([](double x, double y, double) {
            return 0.25 + 0.2 * std::sin(12.566 * y); });

        PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
        for (ScalarField* f : {&phi0, &phia, &phib}) {
            f->allocDevice(); f->uploadAllToDevice();
            bx.applyOnGPU(*f); by.applyOnGPU(*f);
        }

        // ---- Path A: ExprTree / EvalPlan, mu_0 (10 terms, mirrors MPF_AC_DW) ----
        auto treeMu = [&](ScalarField& p, ScalarField& a, ScalarField& b,
                          double ea, double eb) {
            ExprTree t =
                ExprTree(p) * ExprTree(a) * ExprTree(a) * (2.0*W) +
                ExprTree(p) * ExprTree(b) * ExprTree(b) * (2.0*W) +
                ExprTree(p) * expr_grad_dot(a, a) * ( 2.0*ea) +
                ExprTree(a) * expr_grad_dot(p, a) * (-2.0*ea) +
                ExprTree(p) * ExprTree(a) * expr_lap(a) * ( ea) +
                ExprTree(a) * ExprTree(a) * expr_lap(p) * (-ea) +
                ExprTree(p) * expr_grad_dot(b, b) * ( 2.0*eb) +
                ExprTree(b) * expr_grad_dot(p, b) * (-2.0*eb) +
                ExprTree(p) * ExprTree(b) * expr_lap(b) * ( eb) +
                ExprTree(b) * ExprTree(b) * expr_lap(p) * (-eb);
            return t;
        };
        ExprTree t0 = treeMu(phi0, phia, phib, e2a, e2b);
        ExprTree ta = treeMu(phia, phi0, phib, e2a, e2ab);
        ExprTree tb = treeMu(phib, phi0, phia, e2b, e2ab);

        EvalPlan plan = lowerExprTree(t0);

        Equation eqA0(phi0), eqAa(phia), eqAb(phib);
        eqA0.setRHS(t0); eqAa.setRHS(ta); eqAb.setRHS(tb);

        ScalarField muA0(mesh, "muA0", 1), muAa(mesh, "muAa", 1), muAb(mesh, "muAb", 1);
        for (ScalarField* f : {&muA0, &muAa, &muAb}) { f->allocDevice(); }

        double msA  = timeIt([&]{ eqA0.computeRHS(muA0); }, 5, 50);
        double msA3 = timeIt([&]{ eqA0.computeRHS(muA0);
                                  eqAa.computeRHS(muAa);
                                  eqAb.computeRHS(muAb); }, 5, 50);

        // ---- Path B/C: FusedTerm ----
        using namespace PhiX::Fused;
        auto fusedMu = [&](ScalarField& p, ScalarField& a, ScalarField& b,
                           double ea, double eb) {
            return
                fmul(fmul(ffield(p), ffield(a)), ffield(a)) * (2.0*W) +
                fmul(fmul(ffield(p), ffield(b)), ffield(b)) * (2.0*W) +
                fmul(ffield(p), fgrad_dot(a, a)) * ( 2.0*ea) +
                fmul(ffield(a), fgrad_dot(p, a)) * (-2.0*ea) +
                fmul(fmul(ffield(p), ffield(a)), flap(a)) * ( ea) +
                fmul(fmul(ffield(a), ffield(a)), flap(p)) * (-ea) +
                fmul(ffield(p), fgrad_dot(b, b)) * ( 2.0*eb) +
                fmul(ffield(b), fgrad_dot(p, b)) * (-2.0*eb) +
                fmul(fmul(ffield(p), ffield(b)), flap(b)) * ( eb) +
                fmul(fmul(ffield(b), ffield(b)), flap(p)) * (-eb);
        };
        auto f0 = fusedMu(phi0, phia, phib, e2a, e2b);
        auto fa = fusedMu(phia, phi0, phib, e2a, e2ab);
        auto fb = fusedMu(phib, phi0, phia, e2b, e2ab);

        Equation eqB(phi0);
        eqB.setRHS(fuse(f0, phi0));
        ScalarField muB(mesh, "muB", 1);
        muB.allocDevice();
        double msB = timeIt([&]{ eqB.computeRHS(muB); }, 5, 50);

        ScalarField muC0(mesh, "muC0", 1), muCa(mesh, "muCa", 1), muCb(mesh, "muCb", 1);
        for (ScalarField* f : {&muC0, &muCa, &muCb}) { f->allocDevice(); }
        double msC = timeIt([&]{ fuse_multi_compute(phi0, muC0, f0, muCa, fa, muCb, fb); },
                            5, 50);

        // ---- correctness: A vs B on physical cells ----
        eqA0.computeRHS(muA0); eqB.computeRHS(muB);
        muA0.downloadCurrFromDevice(); muB.downloadCurrFromDevice();
        double maxdiff = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                double d = std::fabs(muA0.curr[muA0.index(i, j, 0)]
                                   - muB.curr[muB.index(i, j, 0)]);
                if (d > maxdiff) maxdiff = d;
            }

        const double cells = double(N) * N;
        std::printf("N=%5d  EvalPlan steps=%zu\n", N, plan.steps.size());
        std::printf("  A  EvalPlan 1-field   %8.3f ms  %9.1f Mcells/s\n", msA,  cells/(msA *1e3));
        std::printf("  B  Fused    1-field   %8.3f ms  %9.1f Mcells/s   speedup %4.1fx\n",
                    msB, cells/(msB*1e3), msA/msB);
        std::printf("  A3 EvalPlan 3-field   %8.3f ms  %9.1f Mcells/s\n", msA3, 3*cells/(msA3*1e3));
        std::printf("  C  FusedMulti3        %8.3f ms  %9.1f Mcells/s   speedup %4.1fx\n",
                    msC, 3*cells/(msC*1e3), msA3/msC);
        std::printf("  max|A-B| = %.3e\n\n", maxdiff);
    }
    return 0;
}
