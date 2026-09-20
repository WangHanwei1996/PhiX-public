// ---------------------------------------------------------------------------
// test_expr.cpp — Stage 1: ExprTree node construction, ghostRequired()
// inference, constant folding, and validateGhostRequirements().
//
// This is a CPU-only test (no GPU required).
// ---------------------------------------------------------------------------

#include "equation/Expr.h"
#include "mesh/Mesh.h"
#include "field/ScalarField.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iostream>

using namespace PhiX;

// ---------------------------------------------------------- helper --------
static void require(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        throw std::runtime_error(msg);
    }
}
static void pass(const std::string& msg) {
    std::cout << "PASS: " << msg << "\n";
}

// ---------------------------------------------------------- fixtures -------
static Mesh makeMesh2D() {
    return Mesh::makeUniform2D(CoordSys::CARTESIAN,
                               8, 1.0, 0.0,
                               8, 1.0, 0.0);
}

// =========================================================================
// Test 1: Leaf node properties
// =========================================================================
static void test_leaf() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);

    ExprTree t(f);
    require(t.ghostRequired() == 0, "Leaf ghostRequired should be 0");
    require(t.isLocal() == true,     "Leaf should be Local");
    require(t.repField() == &f,      "Leaf repField should be &f");
    pass("test_leaf");
}

// =========================================================================
// Test 2: ExprScale constant folding
//   (2.0 * tree) * 3.0  should collapse to a single ExprScale(6.0, ...)
// =========================================================================
static void test_scale_folding() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);

    ExprTree t(f);
    ExprTree t2 = t * 2.0;
    ExprTree t6 = t2 * 3.0;

    // The inner node of t6 should be ExprScale with coeff == 6.0
    auto* sc = dynamic_cast<ExprScale*>(t6.node.get());
    require(sc != nullptr, "t * 2.0 * 3.0 should yield ExprScale at root");
    require(std::abs(sc->coeff - 6.0) < 1e-12,
            "Folded coeff should be 6.0, got " + std::to_string(sc->coeff));
    require(sc->child.get() == t.node.get(),
            "Folded child should point to original leaf node");
    pass("test_scale_folding");
}

// =========================================================================
// Test 3: Negation
//   -t  →  ExprScale(-1.0, leaf)
// =========================================================================
static void test_negation() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);

    ExprTree t(f);
    ExprTree nt = -t;

    auto* sc = dynamic_cast<ExprScale*>(nt.node.get());
    require(sc != nullptr,               "Negation should yield ExprScale");
    require(std::abs(sc->coeff + 1.0) < 1e-12, "Negation coeff should be -1.0");
    pass("test_negation");
}

// =========================================================================
// Test 4: Addition — ghostRequired = max of children
// =========================================================================
static void test_addition_ghost() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);
    ScalarField g(mesh, "g", 1);

    ExprTree tf(f);
    ExprTree tg(g);
    ExprTree sum = tf + tg;

    require(sum.ghostRequired() == 0, "Sum of two leaves should have ghostRequired=0");
    require(sum.isLocal() == true,    "Sum of two locals should be Local");
    pass("test_addition_ghost");
}

// =========================================================================
// Test 5: Stencil node — ghostRequired = stencilWidth (child ghost hidden)
// =========================================================================
static void test_stencil_ghost() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);

    ExprTree tl = expr_lap(f);
    require(tl.ghostRequired() == 1, "lap(f) should have ghostRequired=1");
    require(tl.isLocal() == false,   "lap(f) should NOT be Local");
    pass("test_stencil_ghost: lap");

    ExprTree tg = expr_grad(f, 0);
    require(tg.ghostRequired() == 1, "grad(f,0) should have ghostRequired=1");
    require(tg.isLocal() == false,   "grad(f,0) should NOT be Local");
    pass("test_stencil_ghost: grad");
}

// =========================================================================
// Test 6: Stencil inside Add — the Add's ghostRequired = max(1, 0) = 1
// =========================================================================
static void test_stencil_in_sum() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);
    ScalarField g(mesh, "g", 1);

    ExprTree sum = expr_lap(f) + ExprTree(g);
    require(sum.ghostRequired() == 1,  "lap(f)+g should have ghostRequired=1");
    require(sum.isLocal() == false,    "lap(f)+g should NOT be Local (has stencil)");
    pass("test_stencil_in_sum");
}

// =========================================================================
// Test 7: validateGhostRequirements — passes when ghost is sufficient
// =========================================================================
static void test_validate_ok() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", /*ghost=*/1);

    ExprTree tree = expr_lap(f);
    // Should not throw
    validateGhostRequirements(tree);
    pass("test_validate_ok");
}

// =========================================================================
// Test 8: validateGhostRequirements — throws when ghost insufficient
// =========================================================================
static void test_validate_fail() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", /*ghost=*/0);  // zero ghost — invalid for stencil

    ExprTree tree = expr_lap(f);
    bool caught = false;
    try {
        validateGhostRequirements(tree);
    } catch (const std::invalid_argument& e) {
        caught = true;
    }
    require(caught, "validateGhostRequirements should throw for ghost=0 with lap");
    pass("test_validate_fail");
}

// =========================================================================
// Test 9: Multiplication produces ExprMul with correct ghostRequired
// =========================================================================
static void test_mul_ghost() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);
    ScalarField g(mesh, "g", 1);

    ExprTree tf(f);
    ExprTree tg(g);
    ExprTree prod = tf * tg;

    require(prod.ghostRequired() == 0, "f*g should have ghostRequired=0");
    require(prod.isLocal() == true,    "f*g should be Local");

    auto* m = dynamic_cast<ExprMul*>(prod.node.get());
    require(m != nullptr, "f*g should be ExprMul at root");
    pass("test_mul_ghost");
}

// =========================================================================
// Test 10: grad_dot stencil binary
// =========================================================================
static void test_grad_dot() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);
    ScalarField g(mesh, "g", 1);

    ExprTree tgd = expr_grad_dot(f, g);
    require(tgd.ghostRequired() == 1, "grad_dot should have ghostRequired=1");
    require(tgd.isLocal() == false,   "grad_dot should NOT be Local");

    auto* sb = dynamic_cast<ExprStencilBinary*>(tgd.node.get());
    require(sb != nullptr,             "grad_dot should be ExprStencilBinary");
    require(sb->kind == StencilKind::GRAD_DOT, "kind should be GRAD_DOT");
    pass("test_grad_dot");
}

// =========================================================================
// Test 11: isLocal deep tree — stencil buried in product
// =========================================================================
static void test_stencil_in_product() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);
    ScalarField g(mesh, "g", 1);

    // f * lap(g)  →  isLocal=false (has stencil child on right)
    ExprTree tree = ExprTree(f) * expr_lap(g);
    require(tree.isLocal() == false,   "f*lap(g) should NOT be Local");
    require(tree.ghostRequired() == 1, "f*lap(g) ghostRequired should be 1");
    pass("test_stencil_in_product");
}

// =========================================================================
// Test 12: repField propagation through scale
// =========================================================================
static void test_repfield() {
    Mesh mesh = makeMesh2D();
    ScalarField f(mesh, "f", 1);

    ExprTree t = 3.0 * ExprTree(f);
    require(t.repField() == &f, "repField should propagate through ExprScale");
    pass("test_repfield");
}

// =========================================================================
// main
// =========================================================================
int main() {
    try {
        test_leaf();
        test_scale_folding();
        test_negation();
        test_addition_ghost();
        test_stencil_ghost();
        test_stencil_in_sum();
        test_validate_ok();
        test_validate_fail();
        test_mul_ghost();
        test_grad_dot();
        test_stencil_in_product();
        test_repfield();
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll Expr Stage-1 tests passed.\n";
    return 0;
}
