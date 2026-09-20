// ---------------------------------------------------------------------------
// test_fused.cu — Stage 5 tests for FusedTerm compile-time expression fusion.
//
// Verifies:
//   1. ffield: constant-field result matches reference.
//   2. flap: Laplacian via fused kernel matches standard lap(f) term.
//   3. fgrad_dot: |∇f|² via fused kernel matches reference.
//   4. fmul(ffield, ffield): element-wise product matches reference.
//   5. fpw: 1-field pointwise functor (f³) matches reference.
//   6. fpw2: 2-field pointwise functor matches reference.
//   7. fuse() → Equation::setRHS: fused expression through advanceTransient
//      gives same result as standard lap(f).
//   8. fuse_multi_compute: three simultaneous output fields match independent
//      single-output computations (pool-level kernel fusion).
//   9. fuse with explicit CUDA stream: correct result after stream sync.
//
// ---------------------------------------------------------------------------

#include "equation/FusedTerm.h"
#include "equation/Equation.h"
#include "equation/Term.h"
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
using namespace PhiX::Fused;

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

static void fillSmooth(ScalarField& f, double shift = 0.0) {
    int g  = f.ghost;
    int sx = f.storedDims[0], sy = f.storedDims[1];
    int nz = f.mesh.n[2];  // 1 for 2D
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double x = (i + 0.5) * f.mesh.d[0];
        double y = (j + 0.5) * f.mesh.d[1];
        int c = (i+g) + sx*((j+g) + sy*(k+g));  // 3D stored index
        f.curr[c] = std::sin(2*M_PI*x + shift) * std::cos(2*M_PI*y + shift*0.7);
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
    int nz = a.mesh.n[2];  // 1 for 2D
    double d = 0.0;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        int idx = (i+g) + sx*((j+g) + sy*(k+g));  // 3D stored index
        d = std::max(d, std::abs(a.curr[idx] - b.curr[idx]));
    }
    return d;
}

// Zero-out stored buffer on host
static void zeroHost(ScalarField& f) {
    std::fill(f.curr.begin(), f.curr.end(), 0.0);
}

static void zeroDevice(ScalarField& f) {
    if (!f.deviceAllocated()) f.allocDevice();
    CUDA_CHECK_T(cudaMemset(f.d_curr, 0, f.storedSize * sizeof(double)));
}

static int pass_count = 0, fail_count = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s  (violation)\n", msg); } \
    } while(0)

// Reference: compute RHS via standard lap(f) Term into a device buffer.
// f must already have ghost cells filled (BCs applied on GPU).
static void refLapGPU(ScalarField& f, ScalarField& rhs_out,
                      const std::vector<BoundaryCondition*>& bcs) {
    for (auto* bc : bcs) bc->applyOnGPU(f);

    Equation eq(f);
    eq.setRHS(lap(f));
    zeroDevice(rhs_out);
    eq.computeRHS(rhs_out);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_out);
}

// ===========================================================================
// Test 1: ffield leaf node
// ===========================================================================
static void test1_ffield() {
    printf("Test 1: ffield leaf node\n");
    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    // fuse(ffield(f), f) should give d_rhs[c] += f[c]
    ScalarField rhs(m, "rhs", 1);
    zeroDevice(rhs);

    Equation eq(f);
    eq.setRHS(fuse(ffield(f), f));
    eq.computeRHS(rhs);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs);
    d2h(f);

    // Compare d_rhs vs f (both should equal f[c])
    CHECK(maxDiff(rhs, f) < 1e-14, "ffield: rhs == f pointwise");
}

// ===========================================================================
// Test 2: flap matches standard lap(f)
// ===========================================================================
static void test2_flap() {
    printf("Test 2: flap matches standard lap(f)\n");
    int nx = 32, ny = 32;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(f); bc_y.applyOnGPU(f);
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    // Reference via standard lap(f)
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    refLapGPU(f, rhs_ref, bcs);

    // Fused: flap(f)
    ScalarField rhs_fused(m, "rhs_fused", 1);
    zeroDevice(rhs_fused);
    Equation eq_fused(f);
    eq_fused.setRHS(fuse(flap(f), f));
    eq_fused.computeRHS(rhs_fused);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_fused);

    CHECK(maxDiff(rhs_ref, rhs_fused) < 1e-12,
          "flap: fused Laplacian matches standard lap(f)");
}

// ===========================================================================
// Test 3: fmul(ffield, ffield) element-wise product
// ===========================================================================
static void test3_fmul() {
    printf("Test 3: fmul(ffield, ffield) element-wise product\n");
    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1), g(m, "g", 1);
    fillSmooth(f, 0.0);
    fillSmooth(g, 0.5);
    h2d(f); h2d(g);

    // Fused: fmul(ffield(f), ffield(g))
    ScalarField rhs_fused(m, "rhs_fused", 1);
    zeroDevice(rhs_fused);
    Equation eq_fused(f);
    eq_fused.setRHS(fuse(fmul(ffield(f), ffield(g)), f));
    eq_fused.computeRHS(rhs_fused);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_fused);
    d2h(f); d2h(g);

    // CPU reference: f[c] * g[c]  (using 3D stored index)
    ScalarField rhs_ref(m, "rhs_ref", 1);
    {
        int gs = f.ghost, sx = f.storedDims[0], sy = f.storedDims[1];
        int nz = f.mesh.n[2];
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int c = (i+gs) + sx*((j+gs) + sy*(k+gs));
            rhs_ref.curr[c] = f.curr[c] * g.curr[c];
        }
    }

    CHECK(maxDiff(rhs_fused, rhs_ref) < 1e-14,
          "fmul(ffield,ffield): element-wise product correct");
}

// ===========================================================================
// Test 4: fpw (1-field pointwise: f^3)
// ===========================================================================
static void test4_fpw() {
    printf("Test 4: fpw(f, f^3)\n");
    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    ScalarField rhs_fused(m, "rhs_fused", 1);
    zeroDevice(rhs_fused);
    Equation eq(f);
    eq.setRHS(fuse(fpw(f, PHIX_FN (double v) { return v * v * v; }), f));
    eq.computeRHS(rhs_fused);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_fused);
    d2h(f);

    ScalarField rhs_ref(m, "rhs_ref", 1);
    {
        int gs = f.ghost, sx = f.storedDims[0], sy = f.storedDims[1];
        int nz = f.mesh.n[2];
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int c = (i+gs) + sx*((j+gs) + sy*(k+gs));
            rhs_ref.curr[c] = f.curr[c] * f.curr[c] * f.curr[c];
        }
    }

    CHECK(maxDiff(rhs_fused, rhs_ref) < 1e-14, "fpw(f^3): correct");
}

// ===========================================================================
// Test 5: fpw2 (2-field pointwise)
// ===========================================================================
static void test5_fpw2() {
    printf("Test 5: fpw2(f, g, f*g^2)\n");
    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1), g(m, "g", 1);
    fillSmooth(f, 0.1); fillSmooth(g, 0.3);
    h2d(f); h2d(g);

    ScalarField rhs_fused(m, "rhs_fused", 1);
    zeroDevice(rhs_fused);
    Equation eq(f);
    eq.setRHS(fuse(fpw2(f, g, PHIX_FN (double a, double b) { return a * b * b; }), f));
    eq.computeRHS(rhs_fused);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_fused);
    d2h(f); d2h(g);

    ScalarField rhs_ref(m, "rhs_ref", 1);
    {
        int gs = f.ghost, sx = f.storedDims[0], sy = f.storedDims[1];
        int nz = f.mesh.n[2];
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int c = (i+gs) + sx*((j+gs) + sy*(k+gs));
            rhs_ref.curr[c] = f.curr[c] * g.curr[c] * g.curr[c];
        }
    }

    CHECK(maxDiff(rhs_fused, rhs_ref) < 1e-14, "fpw2(f*g^2): correct");
}

// ===========================================================================
// Test 6: composite fused expr: coeff * (flap + fpw2)
// ===========================================================================
static void test6_composite_fused() {
    printf("Test 6: composite fused: coeff*(flap(f) + fpw2(f,f,f-f^3))\n");
    int nx = 32, ny = 32;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(f); bc_y.applyOnGPU(f);

    const double eps2 = 0.04;
    auto expr = flap(f) * eps2
                + fpw2(f, f, PHIX_FN (double a, double b) { return a - a*b*b; });

    ScalarField rhs_fused(m, "rhs_fused", 1);
    zeroDevice(rhs_fused);
    Equation eq(f);
    eq.setRHS(fuse(expr, f));
    eq.computeRHS(rhs_fused);
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(rhs_fused);

    // Reference via two separate standard terms
    ScalarField lap_ref(m, "lap_ref", 1);
    lap_ref.allocDevice();
    refLapGPU(f, lap_ref, {&bc_x, &bc_y});

    // rhs_ref[c] = eps2*lap_f[c] + (f[c] - f[c]^3)
    d2h(f);
    ScalarField rhs_ref(m, "rhs_ref", 1);
    {
        int gs = f.ghost, sx = f.storedDims[0], sy = f.storedDims[1];
        int nz = f.mesh.n[2];
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int c = (i+gs) + sx*((j+gs) + sy*(k+gs));
            rhs_ref.curr[c] = eps2 * lap_ref.curr[c]
                              + f.curr[c] - f.curr[c] * f.curr[c] * f.curr[c];
        }
    }

    CHECK(maxDiff(rhs_fused, rhs_ref) < 1e-12,
          "composite fused expr matches two-term standard reference");
}

// ===========================================================================
// Test 7: fuse_multi_compute (3 outputs, one kernel launch)
// ===========================================================================
static void test7_fuse_multi() {
    printf("Test 7: fuse_multi_compute (3 simultaneous outputs)\n");
    int nx = 16, ny = 16;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1), g(m, "g", 1), h_f(m, "h", 1);
    fillSmooth(f, 0.0); fillSmooth(g, 0.3); fillSmooth(h_f, 0.7);
    h2d(f); h2d(g); h2d(h_f);

    // Three separate output fields
    ScalarField out0(m, "out0", 1), out1(m, "out1", 1), out2(m, "out2", 1);
    out0.allocDevice(); out1.allocDevice(); out2.allocDevice();

    // Expressions: e0 = ffield(f), e1 = fmul(ffield(f), ffield(g)), e2 = ffield(h_f)
    fuse_multi_compute(f,
        out0, ffield(f),
        out1, fmul(ffield(f), ffield(g)),
        out2, ffield(h_f));
    CUDA_CHECK_T(cudaDeviceSynchronize());

    d2h(out0); d2h(out1); d2h(out2);
    d2h(f); d2h(g); d2h(h_f);

    double e0 = 0.0, e1 = 0.0, e2 = 0.0;
    int gs = f.ghost, sx = f.storedDims[0], sy = f.storedDims[1];
    int nz = f.mesh.n[2];
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int c = (i+gs) + sx*((j+gs) + sy*(k+gs));  // 3D stored index
        e0 = std::max(e0, std::abs(out0.curr[c] - f.curr[c]));
        e1 = std::max(e1, std::abs(out1.curr[c] - f.curr[c]*g.curr[c]));
        e2 = std::max(e2, std::abs(out2.curr[c] - h_f.curr[c]));
    }

    CHECK(e0 < 1e-14, "fuse_multi_compute: out0 == f");
    CHECK(e1 < 1e-14, "fuse_multi_compute: out1 == f*g");
    CHECK(e2 < 1e-14, "fuse_multi_compute: out2 == h");
}

// ===========================================================================
// Test 8: fuse() with explicit CUDA stream
// ===========================================================================
static void test8_fuse_explicit_stream() {
    printf("Test 8: fuse() with explicit CUDA stream\n");
    int nx = 32, ny = 32;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    ScalarField f(m, "f", 1);
    fillSmooth(f);
    h2d(f);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    bc_x.applyOnGPU(f); bc_y.applyOnGPU(f);

    // Reference on default stream
    ScalarField rhs_ref(m, "rhs_ref", 1);
    rhs_ref.allocDevice();
    refLapGPU(f, rhs_ref, {&bc_x, &bc_y});

    // Fused on explicit stream
    cudaStream_t s;
    CUDA_CHECK_T(cudaStreamCreate(&s));

    ScalarField rhs_s(m, "rhs_s", 1);
    zeroDevice(rhs_s);
    Equation eq_s(f);
    eq_s.setStream(s);
    eq_s.setRHS(fuse(flap(f), f));
    eq_s.computeRHS(rhs_s);
    CUDA_CHECK_T(cudaStreamSynchronize(s));
    d2h(rhs_s);

    CUDA_CHECK_T(cudaStreamDestroy(s));

    CHECK(maxDiff(rhs_ref, rhs_s) < 1e-12,
          "fuse() on explicit stream: matches default-stream reference");
}

// ===========================================================================
// Test 9: fuse through advanceTransient (multi-step correctness)
// ===========================================================================
static void test9_fuse_transient() {
    printf("Test 9: fuse through advanceTransient (5 steps)\n");
    int nx = 32, ny = 32;
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                 nx, 1.0/nx, 0.0, ny, 1.0/ny, 0.0);
    const double dt = 5e-6;
    const int    N  = 5;

    ScalarField f_std(m, "f_std", 1), f_fused(m, "f_fused", 1);
    fillSmooth(f_std); fillSmooth(f_fused);
    h2d(f_std); h2d(f_fused);

    PeriodicBC bc_x(m.patch("xmin")), bc_y(m.patch("ymin"));
    std::vector<BoundaryCondition*> bcs = {&bc_x, &bc_y};

    Equation eq_std(f_std);
    eq_std.setRHS(lap(f_std));

    Equation eq_fused(f_fused);
    eq_fused.setRHS(fuse(flap(f_fused), f_fused));

    for (int step = 0; step < N; ++step) {
        eq_std.advanceTransient(bcs, dt);
        eq_fused.advanceTransient(bcs, dt);
    }
    CUDA_CHECK_T(cudaDeviceSynchronize());
    d2h(f_std); d2h(f_fused);

    CHECK(maxDiff(f_std, f_fused) < 1e-12,
          "fuse advanceTransient 5 steps: matches standard lap");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== Stage 5 FusedTerm GPU tests ===\n");
    test1_ffield();
    test2_flap();
    test3_fmul();
    test4_fpw();
    test5_fpw2();
    test6_composite_fused();
    test7_fuse_multi();
    test8_fuse_explicit_stream();
    test9_fuse_transient();
    printf("===================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
