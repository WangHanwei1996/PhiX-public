// ---------------------------------------------------------------------------
// test_vector_equation.cu — Stage 6 tests for VectorEquation generalization.
//
// Verifies:
//   1. setStream() / stream() propagate to all component equations.
//   2. registerBC() registers BCs in all component equations (ExprTree path).
//   3. setRHSComponent(c, ExprTree) sets per-component ExprTree RHS.
//   4. computeRHS on VectorField: each component matches scalar reference.
//   5. advanceTransient N steps: vector result matches independent scalar refs.
//   6. setStream(explicit) + computeRHS on VectorField: correct after sync.
//
// GPU test — requires CUDA device.
// ---------------------------------------------------------------------------

#include "equation/VectorEquation.h"
#include "equation/Equation.h"
#include "equation/Expr.h"
#include "equation/FieldOps.inl"
#include "field/VectorField.h"
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
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(_e));                                    \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)

// 3D stored index for cell (i,j,k=0) in a 2D mesh.
static int idx3D(int i, int j, int g, int sx, int sy) {
    return (i+g) + sx*((j+g) + sy*(0+g));
}

static void fillSmooth(ScalarField& f, double shift = 0.0) {
    int g  = f.ghost;
    int sx = f.storedDims[0], sy = f.storedDims[1];
    int nz = f.mesh.n[2];
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double x = (i + 0.5) * f.mesh.d[0];
        double y = (j + 0.5) * f.mesh.d[1];
        int c = (i+g) + sx*((j+g) + sy*(k+g));
        f.curr[c] = std::sin(2*M_PI*x + shift) * std::cos(2*M_PI*y);
    }
}

static void h2d(ScalarField& f) {
    if (!f.deviceAllocated()) f.allocDevice();
    CUDA_CHECK_T(cudaMemcpy(f.d_curr, f.curr.data(),
                            f.storedSize * sizeof(double),
                            cudaMemcpyHostToDevice));
}
static void d2h(ScalarField& f) {
    CUDA_CHECK_T(cudaMemcpy(f.curr.data(), f.d_curr,
                            f.storedSize * sizeof(double),
                            cudaMemcpyDeviceToHost));
}

static double maxDiff(const ScalarField& a, const ScalarField& b) {
    int g  = a.ghost;
    int sx = a.storedDims[0], sy = a.storedDims[1];
    int nz = a.mesh.n[2];
    double d = 0.0;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        int c = (i+g) + sx*((j+g) + sy*(k+g));
        d = std::max(d, std::abs(a.curr[c] - b.curr[c]));
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
// Test 1: setStream / stream propagation
// ===========================================================================
static void test1_stream_propagation() {
    printf("Test 1: setStream/stream propagation to component equations\n");

    int nx = 8, ny = 8;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    VectorField v(m, "v", 2, 1);  // 2-component, ghost=1
    VectorEquation veq(v);

    // Default: stream is nullptr
    CHECK(veq.stream() == nullptr,
          "default stream() == nullptr");

    // setStream propagates to all components
    cudaStream_t s;
    CUDA_CHECK_T(cudaStreamCreate(&s));
    veq.setStream(s);

    CHECK(veq.stream() == s,
          "stream() returns set stream");
    CHECK(veq.componentEquation(0).stream() == s,
          "component[0] stream == s");
    CHECK(veq.componentEquation(1).stream() == s,
          "component[1] stream == s");

    veq.setStream(nullptr);
    CHECK(veq.stream() == nullptr,
          "reset stream to nullptr OK");

    CUDA_CHECK_T(cudaStreamDestroy(s));
}

// ===========================================================================
// Test 2: computeRHS on VectorField — each component matches scalar ref
// ===========================================================================
static void test2_computeRHS_components() {
    printf("Test 2: computeRHS on VectorField — components match scalar refs\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    VectorField v(m, "v", 2, 1);
    fillSmooth(v[0], 0.0);
    fillSmooth(v[1], 0.5);
    h2d(v[0]); h2d(v[1]);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(v[0]); bc_y.applyOnGPU(v[0]);
    bc_x.applyOnGPU(v[1]); bc_y.applyOnGPU(v[1]);
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Vector equation: each component RHS = lap(v[c])
    VectorEquation veq(v);
    RHSExpr rhs0, rhs1;
    rhs0 += lap(v[0]);
    rhs1 += lap(v[1]);
    VectorRHSExpr vrhs(2);
    vrhs[0] = rhs0;
    vrhs[1] = rhs1;
    veq.setRHS(vrhs);

    VectorField rhs_vf(m, "rhs_vf", 2, 1);
    rhs_vf[0].allocDevice();
    rhs_vf[1].allocDevice();
    veq.computeRHS(rhs_vf);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_vf[0]); d2h(rhs_vf[1]);

    // Scalar references
    ScalarField rhs_ref0(m, "rhs_ref0", 1), rhs_ref1(m, "rhs_ref1", 1);
    rhs_ref0.allocDevice(); rhs_ref1.allocDevice();
    {
        Equation eq0(v[0]); eq0.setRHS(lap(v[0]));
        eq0.computeRHS(rhs_ref0);
    }
    {
        Equation eq1(v[1]); eq1.setRHS(lap(v[1]));
        eq1.computeRHS(rhs_ref1);
    }
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ref0); d2h(rhs_ref1);

    CHECK(maxDiff(rhs_vf[0], rhs_ref0) < 1e-13,
          "VectorEquation computeRHS[0] matches scalar lap ref");
    CHECK(maxDiff(rhs_vf[1], rhs_ref1) < 1e-13,
          "VectorEquation computeRHS[1] matches scalar lap ref");
}

// ===========================================================================
// Test 3: registerBC + setRHSComponent — composite ExprTree lap(c0+c1)
//         BC auto-injection via registerBC on VectorEquation.
// ===========================================================================
static void test3_registerBC_and_ExprTree() {
    printf("Test 3: registerBC + setRHSComponent with composite ExprTree\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    VectorField v(m, "v", 2, 1);
    fillSmooth(v[0], 0.0);
    fillSmooth(v[1], 0.4);
    h2d(v[0]); h2d(v[1]);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Build composite ExprTree: lap(v[0] + v[1])
    // The composite child forces BC auto-injection via the bc_map.
    ExprTree c0_plus_c1 = ExprTree(v[0]) + ExprTree(v[1]);
    ExprTree lap_composite(std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1, c0_plus_c1.node));

    // VectorEquation: component 0 RHS = lap(v[0]+v[1]), component 1 = lap(v[0])
    VectorEquation veq(v);
    veq.registerBC(v[0], bcs);  // auto-inject BCs for v[0] in all components
    veq.registerBC(v[1], bcs);  // auto-inject BCs for v[1] in all components

    veq.setRHSComponent(0, lap_composite);

    // Component 1 also uses composite child: lap(v[0] + (-1)*v[1]) = lap(v[0]) - lap(v[1])
    ExprTree c0_minus_c1 = ExprTree(v[0]) + ExprTree(v[1]) * (-1.0);
    ExprTree lap_diff(std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, 1, c0_minus_c1.node));
    veq.setRHSComponent(1, lap_diff);

    VectorField rhs_vf(m, "rhs_vf", 2, 1);
    rhs_vf[0].allocDevice();
    rhs_vf[1].allocDevice();
    veq.computeRHS(rhs_vf);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_vf[0]); d2h(rhs_vf[1]);

    // Reference: apply BCs manually, then lap(v[0])+lap(v[1]) and lap(v[0])
    bc_x.applyOnGPU(v[0]); bc_y.applyOnGPU(v[0]);
    bc_x.applyOnGPU(v[1]); bc_y.applyOnGPU(v[1]);

    ScalarField rhs_ref0(m, "rhs_ref0", 1), rhs_ref1(m, "rhs_ref1", 1);
    rhs_ref0.allocDevice(); rhs_ref1.allocDevice();
    {
        // lap(v[0]+v[1]) = lap(v[0]) + lap(v[1]) by linearity
        Equation eq0(v[0]);
        RHSExpr r0;
        r0 += lap(v[0]);
        r0 += lap(v[1]);
        eq0.setRHS(r0);
        eq0.computeRHS(rhs_ref0);
    }
    {
        // lap(v[0]-v[1]) = lap(v[0]) - lap(v[1]) by linearity
        Equation eq1(v[0]);
        RHSExpr r1;
        r1 += lap(v[0]);
        r1 += lap(v[1], -1.0);
        eq1.setRHS(r1);
        eq1.computeRHS(rhs_ref1);
    }
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ref0); d2h(rhs_ref1);

    CHECK(maxDiff(rhs_vf[0], rhs_ref0) < 1e-11,
          "registerBC + composite ExprTree lap(c0+c1) matches lap(c0)+lap(c1)");
    CHECK(maxDiff(rhs_vf[1], rhs_ref1) < 1e-11,
          "registerBC + composite ExprTree lap(c0-c1) matches lap(c0)-lap(c1)");
}

// ===========================================================================
// Test 4: advanceTransient N steps — matches independent scalar equations
// ===========================================================================
static void test4_advanceTransient() {
    printf("Test 4: advanceTransient N steps\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    const double dt = 5e-6;
    const int    N  = 5;

    // Vector equation
    VectorField v_vec(m, "v_vec", 2, 1);
    fillSmooth(v_vec[0], 0.0); fillSmooth(v_vec[1], 0.3);
    h2d(v_vec[0]); h2d(v_vec[1]);

    // Scalar references
    ScalarField f0(m, "f0", 1), f1(m, "f1", 1);
    fillSmooth(f0, 0.0); fillSmooth(f1, 0.3);
    h2d(f0); h2d(f1);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Vector equation: lap on each component
    VectorEquation veq(v_vec);
    RHSExpr r0, r1;
    r0 += lap(v_vec[0]);
    r1 += lap(v_vec[1]);
    VectorRHSExpr vrhs(2);
    vrhs[0] = r0; vrhs[1] = r1;
    veq.setRHS(vrhs);

    // Scalar references
    Equation eq0(f0), eq1(f1);
    eq0.setRHS(lap(f0));
    eq1.setRHS(lap(f1));

    for (int step = 0; step < N; ++step) {
        veq.advanceTransient(bcs, dt);
        eq0.advanceTransient(bcs, dt);
        eq1.advanceTransient(bcs, dt);
    }
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(v_vec[0]); d2h(v_vec[1]);
    d2h(f0); d2h(f1);

    CHECK(maxDiff(v_vec[0], f0) < 1e-13,
          "advanceTransient[0] matches scalar reference");
    CHECK(maxDiff(v_vec[1], f1) < 1e-13,
          "advanceTransient[1] matches scalar reference");
}

// ===========================================================================
// Test 5: setStream + computeRHS with explicit stream
// ===========================================================================
static void test5_setStream_computeRHS() {
    printf("Test 5: setStream + computeRHS (explicit stream)\n");

    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    VectorField v(m, "v", 2, 1);
    fillSmooth(v[0], 0.0); fillSmooth(v[1], 0.6);
    h2d(v[0]); h2d(v[1]);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(v[0]); bc_y.applyOnGPU(v[0]);
    bc_x.applyOnGPU(v[1]); bc_y.applyOnGPU(v[1]);

    // Reference on default stream
    VectorEquation veq_ref(v);
    RHSExpr r0, r1;
    r0 += lap(v[0]); r1 += lap(v[1]);
    VectorRHSExpr vrhs_ref(2);
    vrhs_ref[0] = r0; vrhs_ref[1] = r1;
    veq_ref.setRHS(vrhs_ref);

    VectorField rhs_ref(m, "rhs_ref", 2, 1);
    rhs_ref[0].allocDevice(); rhs_ref[1].allocDevice();
    veq_ref.computeRHS(rhs_ref);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_ref[0]); d2h(rhs_ref[1]);

    // Same on explicit stream
    cudaStream_t s;
    CUDA_CHECK_T(cudaStreamCreate(&s));

    VectorEquation veq_s(v);
    VectorRHSExpr vrhs_s(2);
    vrhs_s[0] = r0; vrhs_s[1] = r1;
    veq_s.setRHS(vrhs_s);
    veq_s.setStream(s);

    VectorField rhs_s(m, "rhs_s", 2, 1);
    rhs_s[0].allocDevice(); rhs_s[1].allocDevice();
    veq_s.computeRHS(rhs_s);
    CUDA_CHECK_T(cudaStreamSynchronize(s));
    d2h(rhs_s[0]); d2h(rhs_s[1]);

    CUDA_CHECK_T(cudaStreamDestroy(s));

    CHECK(maxDiff(rhs_ref[0], rhs_s[0]) < 1e-13,
          "setStream: computeRHS[0] on explicit stream matches default");
    CHECK(maxDiff(rhs_ref[1], rhs_s[1]) < 1e-13,
          "setStream: computeRHS[1] on explicit stream matches default");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== Stage 6 VectorEquation GPU tests ===\n");
    test1_stream_propagation();
    test2_computeRHS_components();
    test3_registerBC_and_ExprTree();
    test4_advanceTransient();
    test5_setStream_computeRHS();
    printf("=========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
