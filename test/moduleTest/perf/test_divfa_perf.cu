// ---------------------------------------------------------------------------
// module_divfa_perf — P-4 acceptance: the DSL φ-equation RHS built from
// divFaceAssembled + pw2 must reach ≥ 70% of a hand-fused single kernel.
//
// Workload: EFKP-style anisotropic phase-field RHS on a 2048×2048 grid
// (per-cell θ lookup anisotropy, exactly the PoolSectionGPU k_dphi hot
// pattern with face-interpolated θ):
//
//   rhs = ∇·( W² a(ψ) [ a ∇φ + a′(ψ) ∇φ⊥ ] )  +  (1−φ²)(φ − λ(1−φ²)(U+θ̃))
//
//   hand-written : one kernel, faces in registers  (the 57.75 ms/step style)
//   DSL          : Equation::computeRHS with
//                  divFaceAssembled(phi, theta, aniso) + pw(phi, U, bulk)
//                  (zero-init + 2 accumulating kernels — the honest cost)
//
// Numerical agreement is asserted first (tol 1e-10) so both sides provably
// run the same math; then best-of-3 × 50-iteration timings are compared.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/Term.h"
#include "field/ScalarField.h"
#include "operators/FaceOps.h"
#include "mesh/Mesh.h"
#include "perf/Perf.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

#define CUDA_REQUIRE(call)                                                   \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        require(_e == cudaSuccess,                                           \
                std::string("CUDA: ") + cudaGetErrorString(_e));             \
    } while (0)

// ---- model constants (EFKP-flavoured, values irrelevant to the timing) ----
static constexpr double kEps4 = 0.05, kW0sq = 1.44, kLam = 10.8, kDrv = 0.15;

struct AnisoFlux {
    __host__ __device__ double operator()(int ax, double gx, double gy,
                                          double, double, double th) const {
        const double psi = atan2(gy, gx + 1e-30) - th;
        const double a   = 1.0 + kEps4 * cos(4.0 * psi);
        const double s4  = 4.0 * kEps4 * sin(4.0 * psi);
        return (ax == 0) ? kW0sq * a * (a * gx + s4 * gy)
                         : kW0sq * a * (a * gy - s4 * gx);
    }
};

struct BulkTerm {
    __host__ __device__ double operator()(double p, double u) const {
        const double om = 1.0 - p * p;
        return om * (p - kLam * om * (u + kDrv));
    }
};

// ---- hand-fused single kernel (the k_dphi pattern, θ face-interpolated) ----
__global__ void k_hand_rhs(const double* __restrict__ phi,
                           const double* __restrict__ U,
                           const double* __restrict__ theta,
                           double* __restrict__ rhs,
                           int nx, int ny, int sx, int sy, int g,
                           double invdx, double invdy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int c = (i + g) + sx * ((j + g) + sy * g);

    const double pC = phi[c];
    const double pE = phi[c + 1],  pW = phi[c - 1];
    const double pN = phi[c + sx], pS = phi[c - sx];
    const double pNE = phi[c + 1 + sx], pNW = phi[c - 1 + sx];
    const double pSE = phi[c + 1 - sx], pSW = phi[c - 1 - sx];

    AnisoFlux flux;
    // East face
    double gxE = (pE - pC) * invdx;
    double gyE = 0.25 * invdy * (pNE + pN - pSE - pS);
    const double jxE = flux(0, gxE, gyE, 0.0, 0.0,
                            0.5 * (theta[c] + theta[c + 1]));
    // West face
    double gxW = (pC - pW) * invdx;
    double gyW = 0.25 * invdy * (pN + pNW - pS - pSW);
    const double jxW = flux(0, gxW, gyW, 0.0, 0.0,
                            0.5 * (theta[c - 1] + theta[c]));
    // North face
    double gyN = (pN - pC) * invdy;
    double gxN = 0.25 * invdx * (pNE + pE - pNW - pW);
    const double jyN = flux(1, gxN, gyN, 0.0, 0.0,
                            0.5 * (theta[c] + theta[c + sx]));
    // South face
    double gyS = (pC - pS) * invdy;
    double gxS = 0.25 * invdx * (pE + pSE - pW - pSW);
    const double jyS = flux(1, gxS, gyS, 0.0, 0.0,
                            0.5 * (theta[c - sx] + theta[c]));

    const double div = (jxE - jxW) * invdx + (jyN - jyS) * invdy;
    rhs[c] = div + BulkTerm{}(pC, U[c]);
}

template<typename F>
static void fillPeriodic(ScalarField& f, F fn) {
    const int nx = f.mesh.n[0], ny = f.mesh.n[1];
    const int g = f.ghost;
    for (int j = -g; j < ny + g; ++j)
    for (int i = -g; i < nx + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            fn((i + nx) % nx, (j + ny) % ny);
    if (!f.deviceAllocated()) f.allocDevice();
    f.uploadAllToDevice();
}

int main() {
    const int nx = 2048, ny = 2048;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 1e-7, 0.0,
                                    ny, 1e-7, 0.0);

    ScalarField phi(mesh, "phi", 1), U(mesh, "U", 1), theta(mesh, "theta", 1);
    fillPeriodic(phi, [nx](int i, int j) {
        return std::tanh(0.05 * (i - nx / 2)) + 0.05 * std::sin(0.02 * j);
    });
    fillPeriodic(U, [](int i, int j) {
        return 0.1 * std::sin(0.013 * i) * std::cos(0.017 * j);
    });
    fillPeriodic(theta, [](int i, int j) {
        return 0.5 * std::sin(0.007 * i + 0.011 * j);
    });

    ScalarField rhsHand(mesh, "rhsHand", 1), rhsDsl(mesh, "rhsDsl", 1);
    rhsHand.allocDevice(); rhsDsl.allocDevice();

    const double invdx = 1.0 / mesh.d[0], invdy = 1.0 / mesh.d[1];
    const dim3 blk(32, 8);
    const dim3 grd((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);

    Equation eq(phi);
    eq.setRHS(divFaceAssembled(phi, theta, AnisoFlux{})
              + pw(phi, U, BulkTerm{}));

    // ---- numerical agreement first ----
    k_hand_rhs<<<grd, blk>>>(phi.d_curr, U.d_curr, theta.d_curr,
                             rhsHand.d_curr, nx, ny, phi.storedDims[0],
                             phi.storedDims[1], phi.ghost, invdx, invdy);
    eq.computeRHS(rhsDsl);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    rhsHand.downloadCurrFromDevice();
    rhsDsl.downloadCurrFromDevice();
    double maxDiff = 0.0, maxAbs = 0.0;
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j));
        maxDiff = std::max(maxDiff,
                           std::fabs(rhsHand.curr[c] - rhsDsl.curr[c]));
        maxAbs  = std::max(maxAbs, std::fabs(rhsHand.curr[c]));
    }
    require(maxAbs > 1.0, "workload degenerate (all-zero RHS)");
    require(maxDiff <= 1e-10 * maxAbs,
            "hand vs DSL numerical mismatch: " + std::to_string(maxDiff));

    // ---- timings: best of 3 runs × 50 iterations each ----
    const int iters = 50;
    auto timeHand = [&]() {
        perf::CudaEventTimer t;
        t.start();
        for (int it = 0; it < iters; ++it)
            k_hand_rhs<<<grd, blk>>>(phi.d_curr, U.d_curr, theta.d_curr,
                                     rhsHand.d_curr, nx, ny,
                                     phi.storedDims[0], phi.storedDims[1],
                                     phi.ghost, invdx, invdy);
        return t.stopMs() / iters;
    };
    auto timeDsl = [&]() {
        perf::CudaEventTimer t;
        t.start();
        for (int it = 0; it < iters; ++it)
            eq.computeRHS(rhsDsl);      // includes its own zero-init
        return t.stopMs() / iters;
    };

    timeHand(); timeDsl();   // warmup
    double hand = 1e30, dsl = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        hand = std::min(hand, timeHand());
        dsl  = std::min(dsl,  timeDsl());
    }

    const double ratio = hand / dsl;
    std::printf("module_divfa_perf: 2048x2048, hand %.3f ms  DSL %.3f ms  "
                "DSL/hand = %.0f%%\n", hand, dsl, 100.0 * ratio);
    require(ratio >= 0.70,
            "P-4 acceptance failed: DSL reaches only "
            + std::to_string(100.0 * ratio) + "% of hand-written");

    std::printf("module_divfa_perf: PASSED\n");
    return 0;
}
