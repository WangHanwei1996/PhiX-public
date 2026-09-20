// ---------------------------------------------------------------------------
// module_misuse — API-misuse regression: the formerly SILENT failure modes
// (wrong coordinate system, stale/null fused pointers, ghost bypass, mesh
// mismatch, dt <= 0, foreign BC patches, RK4 swap safety) must now throw
// structured PhixError subclasses; the valid equivalents must still work.
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "core/Error.h"
#include "boundary/BCBatch.h"
#include "boundary/NoFluxBC.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "equation/Expr.h"
#include "equation/FusedTerm.h"
#include "equation/Term.h"
#include "equation/VectorEquation.h"
#include "field/FaceField.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "mesh/Mesh.h"
#include "operators/FaceOps.h"
#include "solver/Solver.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using namespace PhiX;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

template <class E, class Fn>
static bool throwsE(Fn fn) {
    try { fn(); }
    catch (const E&) { return true; }
    catch (...)      { return false; }
    return false;
}

template <class Fn>
static bool noThrow(Fn fn) {
    try { fn(); }
    catch (...) { return false; }
    return true;
}

static void test1_cartesian_only_operators() {   // S1
    printf("[1] non-Cartesian meshes rejected by differential operators\n");
    Mesh cyl = Mesh::makeUniform2D(CoordSys::CYLINDRICAL, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField f(cyl, "f", 1);

    CHECK(throwsE<ValidationError>([&] { lap(f); }),        "lap() rejects CYLINDRICAL");
    CHECK(throwsE<ValidationError>([&] { grad(f, 0); }),    "grad() rejects CYLINDRICAL");
    CHECK(throwsE<ValidationError>([&] { grad_dot(f, f); }),"grad_dot() rejects CYLINDRICAL");
    CHECK(throwsE<ValidationError>([&] { expr_lap(f); }),   "expr_lap() rejects CYLINDRICAL");

    // Escape hatch: pointwise terms are coordinate-agnostic and stay usable.
    CHECK(noThrow([&] { pw(f, PHIX_FN(double v) { return v * v; }); }),
          "pw() still works on a CYLINDRICAL mesh (escape hatch)");
}

static void test2_fused_null_capture() {   // S3
    printf("[2] fused DSL binds storage at execution time\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField f(m, "f", 1), g(m, "g", 1), out(m, "out", 1);
    Term late = Fused::fuse(Fused::ffield(f), f);
    CHECK(noThrow([&] { Fused::ffield(f); }), "ffield can be built before allocation");
    CHECK(noThrow([&] { Fused::fpw2(f, g, PHIX_FN(Real a, Real b) { return a + b; }); }),
          "pointwise nodes retain field identity before allocation");
    out.allocDevice();
    Equation eq(out);
    eq.setRHS(late);
    CHECK(throwsE<std::runtime_error>([&] { eq.computeRHS(out); }),
          "GPU execution rejects an unallocated input");

    f.fill(1.0); g.fill(2.0);
    f.allocDevice(); f.uploadAllToDevice();
    g.allocDevice(); g.uploadAllToDevice();
    CHECK(noThrow([&] { Fused::ffield(f); }),  "ffield() works once on device");
    CHECK(noThrow([&] { Fused::flap(f); }),    "flap() works once on device");
    CHECK(noThrow([&] {
              Fused::fpw2(f, g, PHIX_FN(Real a, Real b) { return a + b; });
          }),
          "fpw2() works once on device");
}

static void test3_ghost_bypass_closed() {   // S4
    printf("[3] ghost-0 fields rejected on composite stencil paths\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField g0(m, "g0", 0), g1(m, "g1", 1);

    CHECK(throwsE<ValidationError>([&] { grad_dot(g0, g0); }),
          "grad_dot on ghost-0 fields throws (was silent OOB)");
    CHECK(noThrow([&] { grad_dot(g1, g1); }), "grad_dot on ghost-1 fields works");

    // setRHS(ExprTree) now runs validateGhostRequirements (std type kept)
    Equation eq(g0, "eq");
    CHECK(throwsE<std::invalid_argument>([&] { eq.setRHS(expr_lap(g0)); }),
          "setRHS(ExprTree) validates leaf ghosts (was skipped)");
}

static void test4_divface_validation() {   // S5 + rhsGhost
    printf("[4] divFace flux consistency\n");
    Mesh mA = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    Mesh mB = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.2, 0.0, 8, 0.2, 0.0);
    FaceField fx(mA, 0, "fx", 1), fy(mA, 1, "fy", 1);
    FaceField fyB(mB, 1, "fyB", 1);

    CHECK(noThrow([&] { divFace(&fx, &fy, nullptr); }),
          "matched fluxes accepted");
    CHECK(throwsE<ValidationError>([&] { divFace(&fx, &fyB, nullptr); }),
          "fluxes on different meshes rejected (was silent wrong divergence)");
    CHECK(throwsE<ValidationError>([&] { divFace(&fy, &fx, nullptr); }),
          "swapped x/y flux arguments rejected via normalAxis");

    // rhsGhost: divFace assumes the rhs uses the face ghost width
    ScalarField u(mA, "u", 2);
    u.fill(0.0); u.allocDevice(); u.uploadAllToDevice();
    ScalarField rhs2(mA, "rhs2", 2);
    rhs2.allocDevice();
    fx.allocDevice(); fx.uploadToDevice();
    fy.allocDevice(); fy.uploadToDevice();
    Equation eq(u, "eq");
    eq.setRHS(divFace(&fx, &fy, nullptr));
    CHECK(throwsE<ValidationError>([&] { eq.computeRHS(rhs2); }),
          "divFace into a ghost-2 rhs rejected (was memory corruption)");
}

static void test5_dt_validation() {   // S6
    printf("[5] dt <= 0 rejected across the solver layer\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 16, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(0.0); u.allocDevice(); u.uploadAllToDevice();
    Equation eq(u, "eq");
    eq.setRHS(lap(u));

    CHECK(throwsE<ValidationError>([&] { Solver s(eq, {}, 0.0); }),
          "Solver ctor rejects dt = 0");
    CHECK(throwsE<ValidationError>([&] { Solver s(eq, {}, -1e-3); }),
          "Solver ctor rejects dt < 0");
    CHECK(throwsE<ValidationError>([&] {
              EquationSystem sys(0.0, TimeScheme::EULER);
          }),
          "EquationSystem ctor rejects dt = 0");
    CHECK(throwsE<ValidationError>([&] { eq.advanceTransient({}, -0.5); }),
          "advanceTransient rejects dt < 0 (was silent backward integration)");
    CHECK(noThrow([&] { Solver s(eq, {}, 1e-3); }), "valid dt accepted");
}

static void test6_empty_mesh_funnel() {   // S9
    printf("[6] default-constructed mesh rejected at field creation\n");
    Mesh bad;
    CHECK(throwsE<ValidationError>([&] { ScalarField f(bad, "f", 1); }),
          "ScalarField on an invalid mesh throws (was silent empty run)");
}

static void test7_mesh_identity() {   // S10
    printf("[7] equal-n different-geometry meshes rejected in the DSL\n");
    Mesh mA = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    Mesh mB = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.2, 0.0, 8, 0.2, 0.0);
    ScalarField a(mA, "a", 1), b(mB, "b", 1);

    CHECK(throwsE<ValidationError>([&] {
              pw(a, b, PHIX_FN(double x, double y) { return x * y; });
          }),
          "pw(f1,f2) rejects same-n different-spacing (was n[]-only check)");
    CHECK(throwsE<ValidationError>([&] { ExprTree(a) + ExprTree(b); }),
          "ExprTree operator+ rejects mesh mismatch at composition");
    CHECK(throwsE<ValidationError>([&] { ExprTree(a) * ExprTree(b); }),
          "ExprTree operator* rejects mesh mismatch at composition");
    CHECK(noThrow([&] { ExprTree(a) + expr_lap(a); }),
          "same-mesh composition still works");
}

static void test8_foreign_bc_patch() {   // S12
    printf("[8] BC patch must belong to the field's mesh\n");
    Mesh mA = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    Mesh mB = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField f(mA, "f", 1);
    f.fill(0.0); f.allocDevice(); f.uploadAllToDevice();

    NoFluxBC own(mA.facePatch(Axis::X, Side::LOW));
    NoFluxBC foreign(mB.facePatch(Axis::X, Side::LOW));

    BCBatch okBatch;
    CHECK(noThrow([&] { okBatch.build(f, {&own}); }),
          "own-mesh patch accepted");
    BCBatch badBatch;
    CHECK(throwsE<ValidationError>([&] { badBatch.build(f, {&foreign}); }),
          "foreign-mesh patch rejected (was OOB ghost writes)");
}

static void test9_iso_grad_warns() {   // S14
    printf("[9] iso_grad 3D fallback warns instead of silently degrading\n");
    Mesh m3 = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                  4, 0.1, 0.0, 4, 0.1, 0.0, 4, 0.1, 0.0);
    ScalarField f(m3, "f", 1);

    std::ostringstream oss;
    std::streambuf* old = std::cerr.rdbuf(oss.rdbuf());
    Term t = iso_grad(f, 2);   // 3D + axis 2 → CD2 fallback
    std::cerr.rdbuf(old);
    CHECK(oss.str().find("iso_grad") != std::string::npos
              && oss.str().find("WARNING") != std::string::npos,
          "3D iso_grad emits a one-time [PhiX] WARNING");
    CHECK(t.ghostRequired >= 1, "fallback Term still valid");
}

static void test10_vector_equation_param() {   // S16
    printf("[10] VectorEquation::param checked accessor\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    VectorField v(m, "v", 2, 1);
    VectorEquation eq(v, "eq");
    eq.params["kappa"] = 2.5;

    CHECK(eq.param("kappa") == 2.5, "existing key returned");
    bool listsKeys = false;
    try { eq.param("missing"); }
    catch (const ValidationError& e) {
        listsKeys = e.hint().find("kappa") != std::string::npos;
    }
    CHECK(listsKeys, "missing key throws and lists available keys");
    CHECK(eq.params.size() == 1, "param() does not insert (map unchanged)");
}

static void test11_rk4_swap_exception_safety() {   // S15
    printf("[11] RK4 pointer swap is exception-safe\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 16, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(1.0); u.allocDevice(); u.uploadAllToDevice();
    Real* origPtr = u.d_curr;

    // A Term whose launcher throws on its SECOND call — i.e. during the k2
    // stage evaluation, after the unknown's d_curr was pointed at scratch.
    Term bomb;
    bomb.gpu_launcher = [n = 0](Real*, double, ScratchPool&) mutable {
        if (++n >= 2) throw std::runtime_error("boom at k2");
    };
    bomb.cpu_kernel = [](Real*, double, ScratchPool&) {};

    Equation eq(u, "eq");
    eq.setRHS(bomb);
    Solver solver(eq, {}, 1e-3, TimeScheme::RK4);

    bool threw = false;
    try { solver.advance(); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw, "stage-2 RHS failure propagates");
    CHECK(u.d_curr == origPtr,
          "unknown.d_curr restored after mid-stage throw (was left aliasing scratch)");
}

static void test12_setrhs_overload_switch() {
    printf("[12] setRHS(RHSExpr) clears a previously set ExprTree plan\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 16, 0.5);
    ScalarField u(m, "u", 1), rhs(m, "rhs", 1);
    u.fill(1.0); u.allocDevice(); u.uploadAllToDevice();
    rhs.allocDevice();

    Equation eq(u, "eq");
    eq.setRHS(expr_lap(u));                 // ExprTree plan: lap(1)=0
    eq.setRHS(RHSExpr(pw(u, PHIX_FN (double v) { return 3.0 * v; })));

    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();
    const double v = rhs.curr[rhs.index(8)];
    CHECK(v > 2.999 && v < 3.001,
          "after switching to RHSExpr the new RHS executes (was stale plan = 0)");
}

int main() {
    printf("=== module_misuse: silent-failure gaps now throw ===\n");
    test1_cartesian_only_operators();
    test2_fused_null_capture();
    test3_ghost_bypass_closed();
    test4_divface_validation();
    test5_dt_validation();
    test6_empty_mesh_funnel();
    test7_mesh_identity();
    test8_foreign_bc_patch();
    test9_iso_grad_warns();
    test10_vector_equation_param();
    test11_rk4_swap_exception_safety();
    test12_setrhs_overload_switch();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
