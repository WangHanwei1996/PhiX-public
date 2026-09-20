// ---------------------------------------------------------------------------
// module_health — default-on per-step health sentinel (HealthCheck) on
// Solver and EquationSystem: NaN detection, blow-up limit, opt-out,
// CPU-path host scan.
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "core/Error.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"
#include "operators/Laplacian.h"
#include "solver/Solver.h"

#include <cstdio>
#include <limits>
#include <string>

using namespace PhiX;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

static void test1_defaults() {
    printf("[1] defaults\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(0.0); u.allocDevice(); u.uploadAllToDevice();
    Equation eq(u, "eq");
    eq.setRHS(lap(u));
    Solver solver(eq, {}, 1e-4);

    CHECK(solver.health.every == 100 && solver.health.maxAbsLimit == 0.0,
          "Solver sentinel defaults: every=100, NaN-only");
    EquationSystem sys(1e-4);
    CHECK(sys.health.every == 100, "EquationSystem sentinel default on");
}

static void test2_solver_nan_sentinel() {
    printf("[2] Solver GPU sentinel\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(0.5); u.allocDevice(); u.uploadAllToDevice();
    Equation eq(u, "eq");
    eq.setRHS(lap(u));
    Solver solver(eq, {}, 1e-4);
    solver.health.every = 1;

    bool ok = true;
    try { for (int s = 0; s < 5; ++s) solver.advance(); }
    catch (...) { ok = false; }
    CHECK(ok, "healthy run passes with every=1");

    u.curr[u.index(7)] = kNaN;
    u.uploadCurrToDevice();
    bool labelled = false;
    try { solver.advance(); }
    catch (const NumericsError& e) {
        labelled = e.message().find("'u'") != std::string::npos
                && e.message().find("step") != std::string::npos;
    }
    CHECK(labelled, "NaN aborts the step with field name + step in the error");

    // Opt-out: same poisoned field, sentinel off -> no throw.
    solver.health.every = 0;
    bool silent = true;
    try { solver.advance(); }
    catch (...) { silent = false; }
    CHECK(silent, "health.every = 0 disables the sentinel");
}

static void test3_blowup_limit() {
    printf("[3] maxAbsLimit fires before NaN\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(10.0);   // finite but "large"
    u.allocDevice(); u.uploadAllToDevice();
    Equation eq(u, "eq");
    eq.setRHS(lap(u));
    Solver solver(eq, {}, 1e-4);
    solver.health.every       = 1;
    solver.health.maxAbsLimit = 5.0;

    bool blewUp = false;
    try { solver.advance(); }
    catch (const NumericsError& e) {
        blewUp = e.message().find("maxAbsLimit") != std::string::npos;
    }
    CHECK(blewUp, "|max| = 10 trips limit 5 while still finite");

    solver.health.maxAbsLimit = 1e6;
    bool ok = true;
    try { solver.advance(); }
    catch (...) { ok = false; }
    CHECK(ok, "raised limit passes again");
}

static void test4_equation_system_sentinel() {
    printf("[4] EquationSystem sentinel\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
    ScalarField a(m, "a", 1), b(m, "b", 1);
    a.fill(0.3); a.allocDevice(); a.uploadAllToDevice();
    b.fill(0.7); b.allocDevice(); b.uploadAllToDevice();
    Equation eqA(a, "eqA"), eqB(b, "eqB");
    eqA.setRHS(lap(a));
    eqB.setRHS(lap(b));

    EquationSystem sys(1e-4);
    sys.health.every = 1;
    sys.add(eqA);
    sys.add(eqB);

    bool ok = true;
    try { for (int s = 0; s < 3; ++s) sys.advance(); }
    catch (...) { ok = false; }
    CHECK(ok, "healthy coupled system passes");

    b.curr[b.index(3)] = kNaN;
    b.uploadCurrToDevice();
    bool namesB = false;
    try { sys.advance(); }
    catch (const NumericsError& e) {
        namesB = e.message().find("'b'") != std::string::npos;
    }
    CHECK(namesB, "sentinel names the offending unknown of the system");

    sys.health.every = 0;
    bool silent = true;
    try { sys.advance(); }
    catch (...) { silent = false; }
    CHECK(silent, "system sentinel opt-out works");
}

static void test5_cpu_path_scans_host() {
    printf("[5] CPU path scans the host copy\n");
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 32, 0.1);
    ScalarField u(m, "u", 1);
    u.fill(0.5); u.allocDevice(); u.uploadAllToDevice();
    Equation eq(u, "eq");
    eq.setRHS(lap(u));
    Solver solver(eq, {}, 1e-4);
    solver.health.every = 1;

    // Poison ONLY the host copy — the device stays clean, so a device scan
    // would miss it; advanceCPU must catch it via the host scan.
    u.curr[u.index(5)] = kNaN;
    bool caught = false;
    try { solver.advanceCPU(); }
    catch (const NumericsError&) { caught = true; }
    CHECK(caught, "advanceCPU sentinel reads host data (device is clean)");
}

int main() {
    printf("=== module_health: default-on solution-health sentinel ===\n");
    test1_defaults();
    test2_solver_nan_sentinel();
    test3_blowup_limit();
    test4_equation_system_sentinel();
    test5_cpu_path_scans_host();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
