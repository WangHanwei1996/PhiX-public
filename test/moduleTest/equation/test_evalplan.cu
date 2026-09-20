// ---------------------------------------------------------------------------
// test_evalplan.cu  — Stage 2 integration tests for lowerExprTree / EvalPlan
//
// Verifies that setRHS(ExprTree) produces numerically identical results to
// setRHS(RHSExpr) built from the existing Term-based API.
//
// Tests:
//   1. expr_lap(f) == lap(f)
//   2. expr_grad(f, 0) == grad(f, 0)
//   3. expr_lap(f)*2.0 + expr_grad(f,1) == 2*lap(f) + grad(f,1)
//   4. expr_grad_dot(f,g) == grad_dot(f,g)
//   5. Scalar-only: expr_lap(f)/(-0.5) == lap(f, -2.0)
//   6. ExprTree negation and subtraction: -expr_lap(f) == lap(f,-1)
//   7. Composite expression on leaf: expr_grad(f,0) + expr_grad(g,0)
//   8. ExprTree * ExprTree (Hadamard) via ExprLeaf
//   9. ExprScalar-only RHS (constant fill)
//  10. setRHS(ExprTree) ghost validation throws on insufficient ghost
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/EvalPlan.h"
#include "equation/Expr.h"
#include "equation/Term.h"
#include "equation/TermPW.inl"
#include "equation/FieldOps.inl"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace PhiX;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Mesh make2DMesh(int nx, int ny) {
    Mesh m;
    m.dim  = 2;
    m.n[0] = nx; m.n[1] = ny; m.n[2] = 1;
    m.d[0] = 1.0 / nx; m.d[1] = 1.0 / ny; m.d[2] = 1.0;
    return m;
}

// Fill physical cells with smooth function f(i,j) = sin(2pi*i*dx)*cos(2pi*j*dy)
static void fillSmooth(ScalarField& f) {
    const Mesh& m = f.mesh;
    int g = f.ghost;
    int sx = f.storedDims[0];
    for (int j = 0; j < m.n[1]; ++j)
    for (int i = 0; i < m.n[0]; ++i) {
        double x = (i + 0.5) * m.d[0];
        double y = (j + 0.5) * m.d[1];
        f.curr[(i+g) + sx*((j+g))] = std::sin(2*M_PI*x) * std::cos(2*M_PI*y);
    }
}

// Max abs difference between physical cells of two stored arrays.
static double maxDiff(const ScalarField& a, const ScalarField& b) {
    assert(a.storedSize == b.storedSize);
    double diff = 0.0;
    int g  = a.ghost;
    int sx = a.storedDims[0];
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        int idx = (i+g) + sx*(j+g);
        diff = std::max(diff, std::abs(a.curr[idx] - b.curr[idx]));
    }
    return diff;
}

// Synchronise device→host for a field (copy d_curr → curr).
static void d2h(ScalarField& f) {
    cudaMemcpy(f.curr.data(), f.d_curr,
               f.storedSize * sizeof(double), cudaMemcpyDeviceToHost);
}

// Utility: allocate field on device and copy from host.
static void h2d(ScalarField& f) {
    if (!f.deviceAllocated()) f.allocDevice();
    cudaMemcpy(f.d_curr, f.curr.data(),
               f.storedSize * sizeof(double), cudaMemcpyHostToDevice);
}

// Run one RHSExpr-based computation and return result on host.
static ScalarField runRHSExpr(const Mesh& m, int ghost,
                               const RHSExpr& rhs_expr,
                               const std::vector<ScalarField*>& src_fields)
{
    ScalarField dummy(m, "dummy_unknown", ghost);
    Equation eq(dummy);
    eq.setRHS(rhs_expr);

    ScalarField rhs_out(m, "rhs_out", ghost);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);
    return rhs_out;
}

// Run one ExprTree-based computation and return result on host.
static ScalarField runExprTree(const Mesh& m, int ghost,
                                const ExprTree& tree,
                                ScalarField& unknown_field)
{
    Equation eq(unknown_field);
    eq.setRHS(tree);

    ScalarField rhs_out(m, "rhs_out_expr", ghost);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);
    return rhs_out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static int pass = 0, fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass; printf("  PASS: %s\n", msg); } \
        else      { ++fail; printf("  FAIL: %s\n", msg); } \
    } while(0)

static void test1_lap() {
    printf("Test 1: expr_lap(f) == lap(f)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    auto ref  = runRHSExpr(m, 1, RHSExpr(lap(f)), {&f});
    auto got  = runExprTree(m, 1, expr_lap(f), f);
    CHECK(maxDiff(ref, got) < 1e-12, "lap numerical match");
}

static void test2_grad() {
    printf("Test 2: expr_grad(f,0) == grad(f,0)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    auto ref = runRHSExpr(m, 1, RHSExpr(grad(f, 0)), {&f});
    auto got = runExprTree(m, 1, expr_grad(f, 0), f);
    CHECK(maxDiff(ref, got) < 1e-12, "grad(f,0) numerical match");
}

static void test3_sum() {
    printf("Test 3: 2*expr_lap(f) + expr_grad(f,1) == 2*lap(f)+grad(f,1)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    RHSExpr ref_expr(lap(f, 2.0));
    ref_expr += grad(f, 1);
    auto ref = runRHSExpr(m, 1, ref_expr, {&f});

    ExprTree tree = expr_lap(f) * 2.0 + expr_grad(f, 1);
    auto got = runExprTree(m, 1, tree, f);
    CHECK(maxDiff(ref, got) < 1e-12, "2*lap+grad sum");
}

static void test4_grad_dot() {
    printf("Test 4: expr_grad_dot(f,g) == grad_dot(f,g)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    ScalarField g(m, "g", 1);
    fillSmooth(f);
    // g(i,j) = cos(2pi*x)*sin(2pi*y)
    {
        int ghost = g.ghost;
        int sx    = g.storedDims[0];
        for (int j = 0; j < m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i) {
            double x = (i + 0.5) * m.d[0];
            double y = (j + 0.5) * m.d[1];
            g.curr[(i+ghost) + sx*(j+ghost)] =
                std::cos(2*M_PI*x) * std::sin(2*M_PI*y);
        }
    }
    h2d(f);
    h2d(g);

    auto ref = runRHSExpr(m, 1, RHSExpr(grad_dot(f, g)), {&f, &g});
    ExprTree tree = expr_grad_dot(f, g);
    Equation eq(f);  // f is the unknown (layout)
    eq.setRHS(tree);
    ScalarField rhs_out(m, "rhs_out", 1);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);
    CHECK(maxDiff(ref, rhs_out) < 1e-12, "grad_dot numerical match");
}

static void test5_coeff() {
    printf("Test 5: expr_lap(f)/(-0.5) == lap(f,-2)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    auto ref = runRHSExpr(m, 1, RHSExpr(lap(f, -2.0)), {&f});
    auto got = runExprTree(m, 1, expr_lap(f) / (-0.5), f);
    CHECK(maxDiff(ref, got) < 1e-12, "coefficient division");
}

static void test6_negation() {
    printf("Test 6: -expr_lap(f) == lap(f,-1)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    auto ref = runRHSExpr(m, 1, RHSExpr(lap(f, -1.0)), {&f});
    auto got = runExprTree(m, 1, -expr_lap(f), f);
    CHECK(maxDiff(ref, got) < 1e-12, "negation of lap");
}

static void test7_two_fields() {
    printf("Test 7: expr_grad(f,0) + expr_grad(g,0)\n");
    Mesh m = make2DMesh(16, 16);
    ScalarField f(m, "f", 1);
    ScalarField g(m, "g", 1);
    fillSmooth(f);
    {
        int ghost = g.ghost, sx = g.storedDims[0];
        for (int j = 0; j < m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i) {
            double x = (i+0.5)*m.d[0], y = (j+0.5)*m.d[1];
            g.curr[(i+ghost)+sx*(j+ghost)] = std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
        }
    }
    h2d(f); h2d(g);

    RHSExpr ref_expr(grad(f, 0));
    ref_expr += grad(g, 0);
    auto ref = runRHSExpr(m, 1, ref_expr, {&f, &g});

    ExprTree tree = expr_grad(f, 0) + expr_grad(g, 0);
    Equation eq(f);
    eq.setRHS(tree);
    ScalarField rhs_out(m, "rhs_out", 1);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);
    CHECK(maxDiff(ref, rhs_out) < 1e-12, "two-field grad sum");
}

static void test8_hadamard() {
    printf("Test 8: ExprLeaf * ExprLeaf (Hadamard)\n");
    Mesh m = make2DMesh(8, 8);
    ScalarField f(m, "f", 1);
    ScalarField g(m, "g", 1);
    {
        int ghost = f.ghost, sx = f.storedDims[0];
        for (int j = 0; j < m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i) {
            double x = (i+0.5)*m.d[0], y = (j+0.5)*m.d[1];
            f.curr[(i+ghost)+sx*(j+ghost)] = 2.0 * x;
            g.curr[(i+ghost)+sx*(j+ghost)] = 3.0 * y;
        }
    }
    h2d(f); h2d(g);

    // Reference via existing Term API: f * g
    auto ref = runRHSExpr(m, 1, RHSExpr(f * g), {&f, &g});

    // ExprTree Hadamard: ExprTree(f) * ExprTree(g)
    ExprTree ft(f), gt(g);
    ExprTree tree = ft * gt;
    Equation eq(f);
    eq.setRHS(tree);
    ScalarField rhs_out(m, "rhs_out", 1);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);
    CHECK(maxDiff(ref, rhs_out) < 1e-12, "Hadamard f*g");
}

static void test9_const_fill() {
    printf("Test 9: ExprScalar constant fill\n");
    Mesh m = make2DMesh(8, 8);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    // f + 3.0  → every physical cell should equal f[idx] + 3.0
    // Build via ExprTree arithmetic: ExprTree(f) + 3.0
    ExprTree tree = ExprTree(f) + 3.0;
    Equation eq(f);
    eq.setRHS(tree);
    ScalarField rhs_out(m, "rhs_out", 1);
    rhs_out.allocDevice();
    eq.computeRHS(rhs_out);
    d2h(rhs_out);

    // Reference: pw identity for f + pw constant 3.0
    RHSExpr ref_expr;
    ref_expr += pw(f, [] __host__ __device__ (double v) { return v; });
    ref_expr += pw(f, [] __host__ __device__ (double)   { return 3.0; });
    auto ref = runRHSExpr(m, 1, ref_expr, {&f});

    CHECK(maxDiff(ref, rhs_out) < 1e-12, "ExprScalar constant fill");
}

static void test10_ghost_validation() {
    printf("Test 10: ghost validation throws for insufficient ghost\n");
    Mesh m = make2DMesh(8, 8);
    ScalarField f(m, "f", 0);  // ghost=0 — insufficient for lap (needs 1)

    bool threw = false;
    try {
        validateGhostRequirements(expr_lap(f));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "validateGhostRequirements throws for ghost=0 under lap");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== EvalPlan Stage 2 GPU tests ===\n");
    test1_lap();
    test2_grad();
    test3_sum();
    test4_grad_dot();
    test5_coeff();
    test6_negation();
    test7_two_fields();
    test8_hadamard();
    test9_const_fill();
    test10_ghost_validation();
    printf("=================================\n");
    printf("Results: %d passed, %d failed\n", pass, fail);
    return (fail == 0) ? 0 : 1;
}
