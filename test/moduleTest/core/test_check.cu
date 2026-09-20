// ---------------------------------------------------------------------------
// module_check — PhiX::check validation API: scalar/mesh/field checks
// (host) + checkFinite / checkFieldHealth (GPU reductions).
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "core/CudaCheck.h"
#include "core/Error.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cstdio>
#include <limits>
#include <string>

using namespace PhiX;
using namespace PhiX::check;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

// True iff fn() throws exactly E (and not something else).
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

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const double kInf = std::numeric_limits<double>::infinity();

static void test1_scalar_checks() {
    printf("[1] scalar checks\n");
    CHECK(noThrow([] { checkPositive(1.0, "dt", "t1"); }),  "checkPositive accepts 1.0");
    CHECK(throwsE<ValidationError>([] { checkPositive(0.0, "dt", "t1"); }),
          "checkPositive rejects 0");
    CHECK(throwsE<ValidationError>([] { checkPositive(-2.0, "dt", "t1"); }),
          "checkPositive rejects negative");
    CHECK(noThrow([] { checkNonNegative(0.0, "w", "t1"); }), "checkNonNegative accepts 0");
    CHECK(throwsE<ValidationError>([] { checkNonNegative(-0.1, "w", "t1"); }),
          "checkNonNegative rejects negative");
    CHECK(noThrow([] { checkRange(0.5, 0.0, 1.0, "phi", "t1"); }),
          "checkRange accepts interior");
    CHECK(throwsE<ValidationError>([] { checkRange(1.5, 0.0, 1.0, "phi", "t1"); }),
          "checkRange rejects outside");
    CHECK(throwsE<ValidationError>([] { checkFiniteScalar(kNaN, "T", "t1"); }),
          "checkFiniteScalar rejects NaN");
    CHECK(throwsE<ValidationError>([] { checkFiniteScalar(kInf, "T", "t1"); }),
          "checkFiniteScalar rejects Inf");
}

static void test2_mesh_checks() {
    printf("[2] mesh checks\n");
    Mesh bad;   // default-constructed: n = 0, no patches
    CHECK(throwsE<ValidationError>([&] { checkMeshValid(bad, "t2"); }),
          "checkMeshValid rejects default-constructed mesh");

    Mesh cart = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    CHECK(noThrow([&] { checkMeshValid(cart, "t2"); }), "checkMeshValid accepts valid mesh");
    CHECK(noThrow([&] { checkCartesian(cart, "t2"); }), "checkCartesian accepts CARTESIAN");

    Mesh cyl = Mesh::makeUniform1D(CoordSys::CYLINDRICAL, 8, 0.1);
    bool msgNamesSystem = false;
    try { checkCartesian(cyl, "lap"); }
    catch (const ValidationError& e) {
        msgNamesSystem =
            std::string(e.message()).find("CYLINDRICAL") != std::string::npos;
    }
    CHECK(msgNamesSystem, "checkCartesian rejects CYLINDRICAL and names it");

    CHECK(noThrow([&] { checkIndex(cart, 0, 0, 0, "t2"); }),  "checkIndex accepts (0,0,0)");
    CHECK(noThrow([&] { checkIndex(cart, 7, 7, 0, "t2"); }),  "checkIndex accepts corner");
    CHECK(throwsE<ValidationError>([&] { checkIndex(cart, 8, 0, 0, "t2"); }),
          "checkIndex rejects i == nx");
    CHECK(throwsE<ValidationError>([&] { checkIndex(cart, 0, -1, 0, "t2"); }),
          "checkIndex rejects negative j");
}

static void test3_same_mesh() {
    printf("[3] mesh-geometry identity\n");
    Mesh mA  = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    Mesh mA2 = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    Mesh mB  = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.2, 0.0, 8, 0.2, 0.0);
    Mesh mC  = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 5.0, 8, 0.1, 0.0);

    CHECK(sameMeshGeometry(mA, mA2), "identical params on distinct objects match");
    CHECK(!sameMeshGeometry(mA, mB), "same n[] but different spacing differ");
    CHECK(!sameMeshGeometry(mA, mC), "same n[]/d[] but shifted origin differ");

    ScalarField a(mA, "a", 1), a2(mA2, "a2", 1), b(mB, "b", 1), g2(mA, "g2", 2);
    CHECK(noThrow([&] { checkSameMesh(a, a2, "t3"); }),
          "checkSameMesh accepts equal geometry + ghost");
    CHECK(throwsE<ValidationError>([&] { checkSameMesh(a, b, "t3"); }),
          "checkSameMesh rejects different spacing (equal n[])");
    CHECK(throwsE<ValidationError>([&] { checkSameMesh(a, g2, "t3"); }),
          "checkSameMesh rejects different ghost");
}

static void test4_device_checks() {
    printf("[4] device / ghost checks\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField f(m, "f", 1);

    CHECK(throwsE<ValidationError>([&] { checkOnDevice(f, "t4"); }),
          "checkOnDevice rejects before allocDevice");
    f.fill(1.0);
    f.allocDevice();
    f.uploadAllToDevice();
    CHECK(noThrow([&] { checkOnDevice(f, "t4"); }), "checkOnDevice accepts after alloc");

    CHECK(noThrow([&] { checkGhost(f, 1, "t4"); }), "checkGhost accepts ghost >= required");
    CHECK(throwsE<ValidationError>([&] { checkGhost(f, 2, "t4"); }),
          "checkGhost rejects ghost < required");
}

static void test5_check_finite_gpu() {
    printf("[5] checkFinite (GPU)\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 16, 0.1, 0.0, 16, 0.1, 0.0);
    ScalarField f(m, "phi", 1);

    CHECK(throwsE<ValidationError>([&] { check::checkFinite(f, "t5"); }),
          "checkFinite on a host-only field is a ValidationError");

    f.fill(1.0);
    f.allocDevice();
    f.uploadAllToDevice();
    CHECK(noThrow([&] { check::checkFinite(f, "t5"); }), "clean field passes");

    f.curr[f.index(3, 4, 0)] = kNaN;
    f.uploadCurrToDevice();
    bool namesField = false;
    try { check::checkFinite(f, "t5"); }
    catch (const NumericsError& e) {
        namesField = std::string(e.message()).find("'phi'") != std::string::npos;
    }
    CHECK(namesField, "NaN cell raises NumericsError naming the field");

    f.curr[f.index(3, 4, 0)] = kInf;
    f.uploadCurrToDevice();
    CHECK(throwsE<NumericsError>([&] { check::checkFinite(f, "t5"); }),
          "Inf cell raises NumericsError");
}

static void test6_field_health_gpu() {
    printf("[6] checkFieldHealth (GPU)\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 16, 0.1, 0.0, 16, 0.1, 0.0);
    ScalarField f(m, "c", 1);
    f.fill(2.0);
    f.allocDevice();
    f.uploadAllToDevice();

    CHECK(noThrow([&] { checkFieldHealth(f, 10, 0.1); }),
          "finite field passes with maxAbsLimit disabled");
    CHECK(noThrow([&] { checkFieldHealth(f, 10, 0.1, 5.0); }),
          "|max| = 2 passes limit 5");
    CHECK(throwsE<NumericsError>([&] { checkFieldHealth(f, 10, 0.1, 1.5); }),
          "|max| = 2 trips limit 1.5 (blow-up before NaN)");

    f.curr[f.index(0, 0, 0)] = kNaN;
    f.uploadCurrToDevice();
    bool labelsStep = false;
    try { checkFieldHealth(f, 42, 0.42); }
    catch (const NumericsError& e) {
        const std::string msg = e.message();
        labelsStep = msg.find("step 42") != std::string::npos
                  && msg.find("'c'") != std::string::npos;
    }
    CHECK(labelsStep, "NaN raises NumericsError labelled with field and step");
}

static void test7_cuda_check_macros() {
    printf("[7] central CUDA_CHECK / PHIX_KERNEL_CHECK\n");
    CHECK(noThrow([] { PHIX_CUDA_CHECK(cudaSuccess); }),
          "PHIX_CUDA_CHECK passes cudaSuccess");
    CHECK(noThrow([] { CUDA_CHECK(cudaSuccess); }),
          "legacy CUDA_CHECK alias forwards to PHIX_CUDA_CHECK");

    bool hasContext = false;
    try { PHIX_CUDA_CHECK(cudaSetDevice(9999)); }
    catch (const DeviceError& e) {
        const std::string w = e.what();
        hasContext = w.find("PhixError[Device]") != std::string::npos
                  && w.find("test_check.cu") != std::string::npos
                  && w.find("cudaSetDevice(9999)") != std::string::npos;
    }
    CHECK(hasContext,
          "failing call raises DeviceError with file:line and expression");

    cudaGetLastError();   // clear any sticky state
    CHECK(noThrow([] { PHIX_KERNEL_CHECK("test7"); }),
          "PHIX_KERNEL_CHECK passes on clean error state");
}

int main() {
    printf("=== module_check: PhiX::check validation API ===\n");
    test1_scalar_checks();
    test2_mesh_checks();
    test3_same_mesh();
    test4_device_checks();
    test5_check_finite_gpu();
    test6_field_health_gpu();
    test7_cuda_check_macros();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
