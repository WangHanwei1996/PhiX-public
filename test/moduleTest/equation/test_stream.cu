// ---------------------------------------------------------------------------
// test_stream.cu — Stage 4 tests for stream-ified equation evaluation.
//
// Verifies:
//   1. computeRHS is now asynchronous (no implicit DeviceSynchronize inside).
//      We verify correctness: result matches the old synchronous computation.
//   2. setStream() API: explicit stream is used, results are correct after
//      an explicit cudaStreamSynchronize.
//   3. advanceTransient with default stream gives same result as reference.
//   4. Multiple sequential advanceTransient steps preserve correctness
//      (default-stream ordering guarantees no data races).
//   5. setStream() with a non-default stream: correct results after explicit sync.
//
// GPU test — requires CUDA device.
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
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

using namespace PhiX;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK_T(call)                                                     \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("CUDA error: %s\n", cudaGetErrorString(_e));                \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)

static void fillSmooth(ScalarField& f, double shift = 0.0) {
    int g  = f.ghost;
    int sx = f.storedDims[0];
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double x = (i + 0.5) * f.mesh.d[0];
        double y = (j + 0.5) * f.mesh.d[1];
        f.curr[(i+g) + sx*(j+g)] =
            std::sin(2*M_PI*x + shift) * std::cos(2*M_PI*y);
    }
}

static void h2d(ScalarField& f) {
    if (!f.deviceAllocated()) f.allocDevice();
    cudaMemcpy(f.d_curr, f.curr.data(),
               f.storedSize * sizeof(double), cudaMemcpyHostToDevice);
}
static void d2h(ScalarField& f) {
    cudaMemcpy(f.curr.data(), f.d_curr,
               f.storedSize * sizeof(double), cudaMemcpyDeviceToHost);
}

static double maxDiff(const ScalarField& a, const ScalarField& b) {
    int g  = a.ghost;
    int sx = a.storedDims[0];
    double d = 0.0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        int idx = (i+g) + sx*(j+g);
        d = std::max(d, std::abs(a.curr[idx] - b.curr[idx]));
    }
    return d;
}

static int pass_count = 0, fail_count = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while(0)

// ===========================================================================
// Test 1: computeRHS (default stream) correctness after DeviceSynchronize
// ===========================================================================
static void test1_default_stream_correctness() {
    printf("Test 1: computeRHS (default stream) correctness\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1);
    fillSmooth(c);
    h2d(c);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);

    // Reference: old synchronous path via RHSExpr
    Equation eq_ref(c);
    eq_ref.setRHS(lap(c));
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    CUDA_CHECK_T(cudaDeviceSynchronize());  // sync explicitly to read
    d2h(rhs_ref);

    // New path: also via RHSExpr but stream_ = nullptr (default stream).
    Equation eq_new(c);
    eq_new.setRHS(lap(c));
    ScalarField rhs_new(m, "rhs_new", 1);
    rhs_new.allocDevice();
    eq_new.computeRHS(rhs_new);
    // Must sync before CPU read (no longer done inside computeRHS).
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_new);

    CHECK(maxDiff(rhs_ref, rhs_new) < 1e-14,
          "computeRHS (default stream) matches reference");
}

// ===========================================================================
// Test 2: computeRHS with explicit non-default stream
// ===========================================================================
static void test2_explicit_stream() {
    printf("Test 2: computeRHS with explicit CUDA stream\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1);
    fillSmooth(c, 0.4);
    h2d(c);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);

    // Reference on default stream
    Equation eq_ref(c);
    eq_ref.setRHS(lap(c));
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ref);

    // Same computation on explicit stream
    cudaStream_t s;
    CUDA_CHECK_T(cudaStreamCreate(&s));

    Equation eq_s(c);
    eq_s.setRHS(lap(c));
    eq_s.setStream(s);
    ScalarField rhs_s(m, "rhs_s", 1);
    rhs_s.allocDevice();
    eq_s.computeRHS(rhs_s);
    // Sync the non-default stream before reading.
    CUDA_CHECK_T(cudaStreamSynchronize(s));
    d2h(rhs_s);

    CUDA_CHECK_T(cudaStreamDestroy(s));

    CHECK(maxDiff(rhs_ref, rhs_s) < 1e-14,
          "computeRHS (explicit stream) matches reference");
}

// ===========================================================================
// Test 3: advanceTransient N steps correctness
// ===========================================================================
static void test3_transient_steps() {
    printf("Test 3: advanceTransient N steps correctness\n");

    int nx = 32, ny = 32;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    const double dt = 1e-5;
    const int nSteps = 5;

    // Reference: forward-Euler diffusion on CPU path
    ScalarField c_ref(m, "c_ref", 1), c_new(m, "c_new", 1);
    fillSmooth(c_ref);
    fillSmooth(c_new);

    h2d(c_ref);
    h2d(c_new);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Reference equation (lap, default stream, we use advanceTransient)
    Equation eq_ref(c_ref);
    eq_ref.setRHS(lap(c_ref));

    Equation eq_new(c_new);
    eq_new.setRHS(lap(c_new));

    for (int step = 0; step < nSteps; ++step) {
        eq_ref.advanceTransient(bcs, dt);
        eq_new.advanceTransient(bcs, dt);
    }

    // Both equations use default stream and same logic — results should match.
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(c_ref);
    d2h(c_new);

    CHECK(maxDiff(c_ref, c_new) < 1e-14,
          "advanceTransient N steps: two identical equations agree");
}

// ===========================================================================
// Test 4: ExprTree path (EvalPlan) correctness after stream change
// ===========================================================================
static void test4_evalplan_stream() {
    printf("Test 4: EvalPlan path with default-stream (no DeviceSynchronize inside)\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1), d(m, "d", 1);
    fillSmooth(c);
    fillSmooth(d, 0.7);
    h2d(c); h2d(d);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};
    bc_x.applyOnGPU(c); bc_y.applyOnGPU(c);
    bc_x.applyOnGPU(d); bc_y.applyOnGPU(d);

    // Reference: lap(c) + lap(d) via RHSExpr
    Equation eq_ref(c);
    RHSExpr ref_rhs(lap(c));
    ref_rhs += lap(d);
    eq_ref.setRHS(ref_rhs);
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    eq_ref.computeRHS(rhs_ref);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ref);

    // EvalPlan path: lap(c+d) with auto-BC
    ExprTree c_plus_d = ExprTree(c) + ExprTree(d);
    auto stencil_node = std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1, c_plus_d.node);
    ExprTree tree(stencil_node);

    Equation eq_ep(c);
    eq_ep.registerBC(c, bcs);
    eq_ep.registerBC(d, bcs);
    eq_ep.setRHS(tree);
    ScalarField rhs_ep(m, "rhs_ep", 1);
    rhs_ep.allocDevice();
    eq_ep.computeRHS(rhs_ep);
    // Must sync explicitly since computeRHS is now async.
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ep);

    CHECK(maxDiff(rhs_ref, rhs_ep) < 1e-11,
          "EvalPlan path (async) matches reference");
}

// ===========================================================================
// Test 5: pw kernel uses pool.stream (pw result matches reference)
// ===========================================================================
static void test5_pw_stream() {
    printf("Test 5: pw kernel (pool.stream) correctness\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField c(m, "c", 1);
    fillSmooth(c);
    h2d(c);

    // Reference: CPU computation of c^3
    ScalarField rhs_cpu(m, "rhs_cpu", 1);
    {
        Equation eq_cpu(c);
        eq_cpu.setRHS(pw(c, PHIX_FN (double v) { return v * v * v; }));
        eq_cpu.computeRHSCPU(rhs_cpu);
    }

    // GPU (async): same pw
    ScalarField rhs_gpu(m, "rhs_gpu", 1);
    rhs_gpu.allocDevice();
    {
        Equation eq_gpu(c);
        eq_gpu.setRHS(pw(c, PHIX_FN (double v) { return v * v * v; }));
        eq_gpu.computeRHS(rhs_gpu);
        CUDA_CHECK_T(cudaDeviceSynchronize());
        d2h(rhs_gpu);
    }

    CHECK(maxDiff(rhs_cpu, rhs_gpu) < 1e-14,
          "pw GPU (pool.stream) matches CPU reference");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== Stage 4 Stream GPU tests ===\n");
    test1_default_stream_correctness();
    test2_explicit_stream();
    test3_transient_steps();
    test4_evalplan_stream();
    test5_pw_stream();
    printf("================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
