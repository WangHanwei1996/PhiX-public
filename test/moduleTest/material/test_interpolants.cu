// ---------------------------------------------------------------------------
// module_interpolants — material/Interpolants.h: analytic properties,
// bitwise identity with the frozen solver copies, host/device consistency.
// ---------------------------------------------------------------------------

#include "material/Interpolants.h"
#include "core/CudaCheck.h"

#include <cstdio>
#include <cmath>

using namespace PhiX::Material;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

// Frozen oracle copies, re-typed from the pre-v3.3 solvers:
//   GFA_FeB.cu h_func/h_prime/g_prime  and  GrandPotential h_of/hp_of.
__host__ __device__ static double oracle_h_quintic(double x) {
    return x*x*x*(6.0*x*x - 15.0*x + 10.0);
}
__host__ __device__ static double oracle_h_quintic_p(double x) {
    return 30.0*x*x*(1.0 - x)*(1.0 - x);
}
__host__ __device__ static double oracle_h_cubic(double x) {
    return x * x * (3.0 - 2.0 * x);
}
__host__ __device__ static double oracle_g_p(double x) {
    return 2.0*x*(1.0 - x)*(1.0 - 2.0*x);
}

__global__ void k_eval(double* out, double x) {
    out[0] = h_cubic(x);
    out[1] = h_cubic_prime(x);
    out[2] = h_quintic(x);
    out[3] = h_quintic_prime(x);
    out[4] = g_dw(x);
    out[5] = g_dw_prime(x);
}

static void test1_endpoints() {
    printf("[1] endpoint / symmetry properties\n");
    CHECK(h_cubic(0.0) == 0.0 && h_cubic(1.0) == 1.0, "h_cubic pins 0/1");
    CHECK(h_quintic(0.0) == 0.0 && h_quintic(1.0) == 1.0, "h_quintic pins 0/1");
    CHECK(h_cubic_prime(0.0) == 0.0 && h_cubic_prime(1.0) == 0.0,
          "h_cubic' vanishes at wells");
    CHECK(h_quintic_prime(0.0) == 0.0 && h_quintic_prime(1.0) == 0.0,
          "h_quintic' vanishes at wells");
    CHECK(h_cubic(0.5) == 0.5 && h_quintic(0.5) == 0.5, "h(1/2) = 1/2");

    bool sym = true;
    for (double x = 0.0; x <= 1.0; x += 0.05)
        sym = sym && std::fabs(h_quintic(x) + h_quintic(1.0 - x) - 1.0) < 1e-14
                  && std::fabs(h_cubic(x)   + h_cubic(1.0 - x)   - 1.0) < 1e-14;
    CHECK(sym, "h(x) + h(1-x) = 1 (both families)");

    CHECK(g_dw(0.0) == 0.0 && g_dw(1.0) == 0.0, "g vanishes at wells");
    CHECK(g_dw_prime(0.0) == 0.0 && g_dw_prime(0.5) == 0.0
              && g_dw_prime(1.0) == 0.0,
          "g' vanishes at 0, 1/2, 1");
    CHECK(g_dw(0.5) == 0.0625, "g(1/2) = 1/16");
}

static void test2_oracle_identity() {
    printf("[2] bitwise identity with frozen solver copies\n");
    bool ok = true;
    for (double x = -0.2; x <= 1.2; x += 0.01) {
        ok = ok && h_quintic(x)       == oracle_h_quintic(x)
                && h_quintic_prime(x) == oracle_h_quintic_p(x)
                && h_cubic(x)         == oracle_h_cubic(x)
                && g_dw_prime(x)      == oracle_g_p(x);
    }
    CHECK(ok, "h_quintic/h_cubic/g' bitwise-match GFA_FeB & GP solver copies");

    bool sel = h_interp(0.3, true)  == h_quintic(0.3)
            && h_interp(0.3, false) == h_cubic(0.3)
            && h_interp_prime(0.7, true)  == h_quintic_prime(0.7)
            && h_interp_prime(0.7, false) == h_cubic_prime(0.7);
    CHECK(sel, "h_interp runtime selector matches the direct forms");
}

static void test3_device_consistency() {
    printf("[3] host/device consistency\n");
    double* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, 6 * sizeof(double)));
    bool ok = true;
    for (double x : {0.0, 0.31, 0.5, 0.77, 1.0}) {
        k_eval<<<1, 1>>>(d, x);
        PHIX_KERNEL_CHECK("test_interpolants");
        double h[6];
        CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
        ok = ok && h[0] == h_cubic(x)       && h[1] == h_cubic_prime(x)
                && h[2] == h_quintic(x)     && h[3] == h_quintic_prime(x)
                && h[4] == g_dw(x)          && h[5] == g_dw_prime(x);
    }
    CUDA_CHECK(cudaFree(d));
    CHECK(ok, "device evaluation bitwise-matches host at sample points");
}

int main() {
    printf("=== module_interpolants: shared h/g polynomials ===\n");
    test1_endpoints();
    test2_oracle_identity();
    test3_device_consistency();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
