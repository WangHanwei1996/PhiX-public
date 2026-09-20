// ---------------------------------------------------------------------------
// test_bc_injection.cu — Stage 3 tests for BC auto-injection.
//
// Verifies that Equation::registerBC() + setRHS(ExprTree) produces the same
// numerical result as the explicit lap(RHSExpr, bcs) Term API when the
// stencil is applied to a composite expression (not a bare leaf).
//
// Scenario (Cahn-Hilliard-style):
//   Let mu_expr = pw(c, c^3 - c) (nonlinear part)
//   Old API:  eq.setRHS(lap(RHSExpr(pw(c, f)), bcs))
//   New API:  eq.registerBC(c, bcs); eq.setRHS(expr_lap(ExprTree(c)*...))
//
// Because ExprPointwise1 is not yet lowerable, we test with a simpler case:
//   lap(c + d)  where c+d is a composite ExprAdd
//   Reference:  lap(c+d) = lap(c) + lap(d)  (linearity)
//
// Tests:
//   1. lap(ExprAdd(c, d)) with auto-BC == lap(c) + lap(d)  (linearity check)
//   2. grad(ExprAdd(c, d), 0) with auto-BC == grad(c,0)+grad(d,0)
//   3. iso_grad(ExprAdd(c,d),0) with auto-BC == iso_grad(c,0)+iso_grad(d,0)
//   4. Missing BC registration throws std::logic_error
//   5. lap(Scale(c)) with auto-BC == 2*lap(c)
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/EvalPlan.h"
#include "equation/Expr.h"
#include "equation/Term.h"
#include "equation/TermPW.inl"
#include "equation/FieldOps.inl"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"
#include "boundary/PeriodicBC.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace PhiX;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fillSmooth(ScalarField& f, double phase_shift = 0.0) {
    int g  = f.ghost;
    int sx = f.storedDims[0];
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double x = (i + 0.5) * f.mesh.d[0];
        double y = (j + 0.5) * f.mesh.d[1];
        f.curr[(i+g) + sx*(j+g)] =
            std::sin(2*M_PI*x + phase_shift) * std::cos(2*M_PI*y);
    }
}

static void d2h(ScalarField& f) {
    cudaMemcpy(f.curr.data(), f.d_curr,
               f.storedSize * sizeof(double), cudaMemcpyDeviceToHost);
}
static void h2d(ScalarField& f) {
    if (!f.deviceAllocated()) f.allocDevice();
    cudaMemcpy(f.d_curr, f.curr.data(),
               f.storedSize * sizeof(double), cudaMemcpyHostToDevice);
}

static double maxDiff(const ScalarField& a, const ScalarField& b) {
    int g  = a.ghost;
    int sx = a.storedDims[0];
    double diff = 0.0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        int idx = (i+g) + sx*(j+g);
        diff = std::max(diff, std::abs(a.curr[idx] - b.curr[idx]));
    }
    return diff;
}

static int pass_count = 0, fail_count = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s  (cond=%s)\n", msg, #cond); } \
    } while(0)

// ---------------------------------------------------------------------------
// Build 2D periodic BCs for a mesh.
// ---------------------------------------------------------------------------
static std::vector<BoundaryCondition*> make2DPeriodicBCs(
        const Mesh& m,
        PeriodicBC& bc_x, PeriodicBC& bc_y)
{
    (void)bc_x; (void)bc_y;  // constructed in-place by caller
    return { &bc_x, &bc_y };
}

// ===========================================================================
// Test 1: lap(c+d) via auto-BC == lap(c) + lap(d)
// ===========================================================================
static void test1_lap_composite() {
    printf("Test 1: lap(ExprAdd(c,d)) with auto-BC == lap(c)+lap(d)\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0,
                                 ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1), d(m, "d", 1);
    fillSmooth(c, 0.0);
    fillSmooth(d, 0.3);
    h2d(c); h2d(d);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Apply BCs so ghost cells are correct for reference computation.
    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);
    bc_x.applyOnGPU(d); bc_y.applyOnGPU(d);

    // Reference: lap(c) + lap(d)  (using Term API)
    Equation eq_ref(c);
    RHSExpr ref_expr(lap(c));
    ref_expr += lap(d);
    eq_ref.setRHS(ref_expr);
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    d2h(rhs_ref);

    // New API: lap(c + d) with auto-BC.
    // ExprAdd of two leaves → composite child for ExprStencil.
    ExprTree c_plus_d = ExprTree(c) + ExprTree(d);
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1,
        c_plus_d.node);
    ExprTree tree(stencil_node);

    Equation eq_new(c);
    eq_new.registerBC(c, bcs);
    eq_new.registerBC(d, bcs);
    eq_new.setRHS(tree);
    ScalarField rhs_new(m, "rhs_new", 1);
    rhs_new.allocDevice();
    eq_new.computeRHS(rhs_new);
    d2h(rhs_new);

    CHECK(maxDiff(rhs_ref, rhs_new) < 1e-11, "lap(c+d) auto-BC == lap(c)+lap(d)");
}

// ===========================================================================
// Test 2: grad(c+d, 0) via auto-BC == grad(c,0) + grad(d,0)
// ===========================================================================
static void test2_grad_composite() {
    printf("Test 2: grad(ExprAdd(c,d),0) with auto-BC == grad(c,0)+grad(d,0)\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0,
                                 ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1), d(m, "d", 1);
    fillSmooth(c, 0.0);
    fillSmooth(d, 0.6);
    h2d(c); h2d(d);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);
    bc_x.applyOnGPU(d); bc_y.applyOnGPU(d);

    // Reference
    Equation eq_ref(c);
    RHSExpr ref_expr(grad(c, 0));
    ref_expr += grad(d, 0);
    eq_ref.setRHS(ref_expr);
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    d2h(rhs_ref);

    // New API
    ExprTree c_plus_d = ExprTree(c) + ExprTree(d);
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::GRAD, /*axis=*/0, 1,
        c_plus_d.node);
    ExprTree tree(stencil_node);

    Equation eq_new(c);
    eq_new.registerBC(c, bcs);
    eq_new.registerBC(d, bcs);
    eq_new.setRHS(tree);
    ScalarField rhs_new(m, "rhs_new", 1);
    rhs_new.allocDevice();
    eq_new.computeRHS(rhs_new);
    d2h(rhs_new);

    CHECK(maxDiff(rhs_ref, rhs_new) < 1e-11, "grad(c+d,0) auto-BC == grad(c,0)+grad(d,0)");
}

// ===========================================================================
// Test 3: iso_grad(c+d, 0) via auto-BC == iso_grad(c,0)+iso_grad(d,0)
// ===========================================================================
static void test3_iso_grad_composite() {
    printf("Test 3: iso_grad(ExprAdd(c,d),0) with auto-BC\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0,
                                 ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1), d(m, "d", 1);
    fillSmooth(c, 0.0);
    fillSmooth(d, 0.9);
    h2d(c); h2d(d);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);
    bc_x.applyOnGPU(d); bc_y.applyOnGPU(d);

    // Reference
    Equation eq_ref(c);
    RHSExpr ref_expr(iso_grad(c, 0));
    ref_expr += iso_grad(d, 0);
    eq_ref.setRHS(ref_expr);
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    d2h(rhs_ref);

    // New API
    ExprTree c_plus_d = ExprTree(c) + ExprTree(d);
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::ISO_GRAD, /*axis=*/0, 1,
        c_plus_d.node);
    ExprTree tree(stencil_node);

    Equation eq_new(c);
    eq_new.registerBC(c, bcs);
    eq_new.registerBC(d, bcs);
    eq_new.setRHS(tree);
    ScalarField rhs_new(m, "rhs_new", 1);
    rhs_new.allocDevice();
    eq_new.computeRHS(rhs_new);
    d2h(rhs_new);

    CHECK(maxDiff(rhs_ref, rhs_new) < 1e-11, "iso_grad(c+d,0) auto-BC");
}

// ===========================================================================
// Test 4: no BC registration at all throws std::logic_error
// ===========================================================================
static void test4_missing_bc_throws() {
    printf("Test 4: no BC registration throws std::logic_error\n");

    int nx = 8, ny = 8;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0,
                                 ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1), d(m, "d", 1);
    fillSmooth(c); fillSmooth(d);
    h2d(c); h2d(d);

    // No registerBC calls — empty bc_map.
    ExprTree c_plus_d = ExprTree(c) + ExprTree(d);
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1,
        c_plus_d.node);
    ExprTree tree(stencil_node);

    bool threw = false;
    try {
        Equation eq(c);
        // No registerBC → bc_map is empty → lowerExprTree should throw
        eq.setRHS(tree);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw, "empty bc_map throws std::logic_error");
}

// ===========================================================================
// Test 5: lap(2.0 * c) via auto-BC == 2 * lap(c)
// ===========================================================================
static void test5_scale_composite() {
    printf("Test 5: lap(2*c) auto-BC == 2*lap(c)\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0,
                                 ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1);
    fillSmooth(c);
    h2d(c);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);

    // Reference: 2*lap(c)
    Equation eq_ref(c);
    eq_ref.setRHS(lap(c, 2.0));
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    d2h(rhs_ref);

    // New API: lap(ExprScale(2, leaf))
    // ExprScale is peeled in the lowering pass — should still work with
    // composite (ExprScale wraps a Leaf → peeled → treated as leaf case)
    // Actually this is the leaf-path (Scale peeled to leaf).
    // Let's test lap(2*c + 0*c) → ExprAdd → composite path.
    // For 2*leaf: use ExprAdd(leaf, leaf) to force composite path.
    ExprTree two_c = ExprTree(c) + ExprTree(c);  // = 2*c via addition
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1, two_c.node);
    ExprTree tree(stencil_node);

    Equation eq_new(c);
    eq_new.registerBC(c, bcs);
    eq_new.setRHS(tree);
    ScalarField rhs_new(m, "rhs_new", 1);
    rhs_new.allocDevice();
    eq_new.computeRHS(rhs_new);
    d2h(rhs_new);

    CHECK(maxDiff(rhs_ref, rhs_new) < 1e-11, "lap(c+c) auto-BC == 2*lap(c)");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== BC Auto-Injection Stage 3 GPU tests ===\n");
    test1_lap_composite();
    test2_grad_composite();
    test3_iso_grad_composite();
    test4_missing_bc_throws();
    test5_scale_composite();
    printf("===========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
