// ---------------------------------------------------------------------------
// module_divfa — divFaceAssembled fused face-flux divergence (v2.45.0, P-4)
//
// Verifies:
//   1. Identity flux (F = ∂f/∂n) reproduces the standard lap(f) Term on a
//      periodic 2D field — all physical cells, incl. boundary rows.
//   2. Per-cell-θ anisotropic flux (the EFKP/k_dphi pattern, 1 aux field)
//      matches the four-step chain faceGradGPU → interpGPU → facePWGPU →
//      divFace on interior cells (the chain's boundary faces use the
//      nearest-cell interp convention; divFaceAssembled uses ghosts).
//   3. CPU fallback == GPU kernel on the 2-aux overload (2D) and in 3D.
//   4. ghost=0 field is rejected; mesh-mismatched aux is rejected.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/Term.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "operators/FaceOps.h"
#include "mesh/Mesh.h"

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

static void requireClose(double a, double b, double tol,
                         const std::string& msg) {
    const double scale = std::max({std::fabs(a), std::fabs(b), 1.0});
    if (std::fabs(a - b) > tol * scale)
        throw std::runtime_error(msg + "  (" + std::to_string(a)
                                 + " vs " + std::to_string(b) + ")");
}

// Fill physical cells from fn(i,j,k), then wrap ALL ghost entries
// periodically (edges + corners) and upload — no BC objects needed.
template<typename F>
static void fillPeriodic(ScalarField& f, F fn) {
    const int nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    const int g = f.ghost;
    for (int k = -g; k < nz + g; ++k)
    for (int j = -g; j < ny + g; ++j)
    for (int i = -g; i < nx + g; ++i) {
        const int ip = (i + nx) % nx, jp = (j + ny) % ny, kp = (k + nz) % nz;
        f.curr[static_cast<std::size_t>(f.index(i, j, k))] = fn(ip, jp, kp);
    }
    if (!f.deviceAllocated()) f.allocDevice();
    f.uploadAllToDevice();
}

// ============================================================================
// 1. identity flux == lap(f)
// ============================================================================
static void testLaplacianIdentity() {
    const int nx = 24, ny = 20;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.25, 0.0,
                                    ny, 0.5, 0.0);
    ScalarField f(mesh, "f", 1);
    fillPeriodic(f, [](int i, int j, int) {
        return std::sin(0.37 * i) * std::cos(0.53 * j) + 0.2 * i - 0.1 * j;
    });

    ScalarField rhsA(mesh, "rhsA", 1), rhsB(mesh, "rhsB", 1);
    rhsA.allocDevice(); rhsB.allocDevice();

    Equation eqA(f);
    eqA.setRHS(lap(f));
    eqA.computeRHS(rhsA);

    Equation eqB(f);
    eqB.setRHS(divFaceAssembled(f,
        [] __host__ __device__ (int ax, double gx, double gy, double gz,
                                double) {
            return (ax == 0) ? gx : (ax == 1) ? gy : gz;
        }));
    eqB.computeRHS(rhsB);
    CUDA_REQUIRE(cudaDeviceSynchronize());

    rhsA.downloadCurrFromDevice();
    rhsB.downloadCurrFromDevice();
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(f.index(i, j));
        requireClose(rhsA.curr[c], rhsB.curr[c], 1e-12,
                     "identity flux != lap(f) at (" + std::to_string(i) + ","
                     + std::to_string(j) + ")");
    }
}

// ============================================================================
// 2. anisotropic flux (1 aux) vs the 4-step chain, interior cells
// ============================================================================
static const double kEps4 = 0.05, kW0sq = 1.7;

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

static void testChainEquivalence() {
    const int nx = 40, ny = 32;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.2, 0.0,
                                    ny, 0.2, 0.0);
    ScalarField phi(mesh, "phi", 1), theta(mesh, "theta", 1),
                gxc(mesh, "gxc", 1), gyc(mesh, "gyc", 1);
    fillPeriodic(phi, [](int i, int j, int) {
        return std::tanh(0.3 * (i - 20)) + 0.1 * std::sin(0.9 * j);
    });
    fillPeriodic(theta, [](int i, int j, int) {
        return 0.4 * std::sin(0.31 * i + 0.7 * j);
    });

    // cell-centred gradients for the chain (periodic, so ghosts are exact)
    const double invdx = 1.0 / mesh.d[0], invdy = 1.0 / mesh.d[1];
    fillPeriodic(gxc, [&](int i, int j, int) {
        const int ip = (i + 1) % nx, im = (i - 1 + nx) % nx;
        return 0.5 * invdx *
               (phi.curr[static_cast<std::size_t>(phi.index(ip, j))] -
                phi.curr[static_cast<std::size_t>(phi.index(im, j))]);
    });
    fillPeriodic(gyc, [&](int i, int j, int) {
        const int jp = (j + 1) % ny, jm = (j - 1 + ny) % ny;
        return 0.5 * invdy *
               (phi.curr[static_cast<std::size_t>(phi.index(i, jp))] -
                phi.curr[static_cast<std::size_t>(phi.index(i, jm))]);
    });

    // ---- chain: faceGrad + interp + facePW(3) + divFace ----
    FaceField gpx(mesh, 0, "gpx"), gpyx(mesh, 0, "gpyx"), thx(mesh, 0, "thx"),
              jx(mesh, 0, "jx");
    FaceField gpy(mesh, 1, "gpy"), gpxy(mesh, 1, "gpxy"), thy(mesh, 1, "thy"),
              jy(mesh, 1, "jy");
    for (FaceField* ff : {&gpx, &gpyx, &thx, &jx, &gpy, &gpxy, &thy, &jy})
        ff->allocDevice();

    faceGradGPU(phi, 0, gpx);
    interpGPU(gyc, 0, gpyx);
    interpGPU(theta, 0, thx);
    facePWGPU(jx, gpx, gpyx, thx,
        [] __host__ __device__ (double px, double py, double th) {
            return AnisoFlux{}(0, px, py, 0.0, 0.0, th);
        });

    faceGradGPU(phi, 1, gpy);
    interpGPU(gxc, 1, gpxy);
    interpGPU(theta, 1, thy);
    facePWGPU(jy, gpy, gpxy, thy,
        [] __host__ __device__ (double py, double px, double th) {
            return AnisoFlux{}(1, px, py, 0.0, 0.0, th);
        });

    ScalarField rhsChain(mesh, "rhsChain", 1), rhsFused(mesh, "rhsFused", 1);
    rhsChain.allocDevice(); rhsFused.allocDevice();

    Equation eqChain(phi);
    eqChain.setRHS(divFace(jx, jy));
    eqChain.computeRHS(rhsChain);

    Equation eqFused(phi);
    eqFused.setRHS(divFaceAssembled(phi, theta, AnisoFlux{}));
    eqFused.computeRHS(rhsFused);
    CUDA_REQUIRE(cudaDeviceSynchronize());

    rhsChain.downloadCurrFromDevice();
    rhsFused.downloadCurrFromDevice();
    for (int j = 1; j < ny - 1; ++j)
    for (int i = 1; i < nx - 1; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j));
        requireClose(rhsChain.curr[c], rhsFused.curr[c], 1e-12,
                     "chain vs fused at (" + std::to_string(i) + ","
                     + std::to_string(j) + ")");
    }
}

// ============================================================================
// 3. CPU fallback == GPU (2-aux overload, 2D + 3D)
// ============================================================================
static void testCpuGpu2Aux(const Mesh& mesh, const std::string& tag) {
    ScalarField f(mesh, "f", 1), a1(mesh, "a1", 1), a2(mesh, "a2", 1);
    fillPeriodic(f,  [](int i, int j, int k) {
        return std::sin(0.4 * i + 0.2 * k) * std::cos(0.3 * j);
    });
    fillPeriodic(a1, [](int i, int j, int k) {
        return 0.5 + 0.3 * std::cos(0.5 * i - 0.2 * j + 0.1 * k);
    });
    fillPeriodic(a2, [](int i, int j, int k) {
        return 0.2 * std::sin(0.7 * j + 0.3 * k);
    });

    // degenerate-mobility-shaped flux: M(a1)·∂f/∂n + a2-weighted cross term
    auto flux = [] __host__ __device__ (int ax, double gx, double gy,
                                        double gz, double fF, double m,
                                        double w) {
        const double gn = (ax == 0) ? gx : (ax == 1) ? gy : gz;
        return m * m * gn + 0.1 * w * fF * (gx + gy + gz);
    };

    Equation eq(f);
    eq.setRHS(divFaceAssembled(f, a1, a2, flux, 1.3));

    ScalarField rhsG(mesh, "rhsG", 1), rhsC(mesh, "rhsC", 1);
    rhsG.allocDevice();
    eq.computeRHS(rhsG);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    rhsG.downloadCurrFromDevice();

    eq.computeRHSCPU(rhsC);

    for (int k = 0; k < mesh.n[2]; ++k)
    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        const std::size_t c = static_cast<std::size_t>(f.index(i, j, k));
        requireClose(rhsG.curr[c], rhsC.curr[c], 1e-12,
                     tag + ": CPU vs GPU mismatch");
    }
}

int main() {
    testLaplacianIdentity();
    testChainEquivalence();

    testCpuGpu2Aux(Mesh::makeUniform2D(CoordSys::CARTESIAN, 32, 0.3, 0.0,
                                       24, 0.4, 0.0), "2D");
    testCpuGpu2Aux(Mesh::makeUniform3D(CoordSys::CARTESIAN, 14, 0.3, 0.0,
                                       12, 0.4, 0.0, 10, 0.5, 0.0), "3D");

    // 4. rejection paths
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 1.0, 0.0,
                                        8, 1.0, 0.0);
        Mesh other = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 1.0, 0.0,
                                         6, 1.0, 0.0);
        ScalarField f0(mesh, "f0", 0), f1(mesh, "f1", 1), fo(other, "fo", 1);
        auto flux = [] __host__ __device__ (int, double gx, double, double,
                                            double) { return gx; };
        bool threw = false;
        try { divFaceAssembled(f0, flux); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "ghost=0 must be rejected");

        threw = false;
        try {
            divFaceAssembled(f1, fo,
                [] __host__ __device__ (int, double gx, double, double,
                                        double, double) { return gx; });
        } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "mesh-mismatched aux must be rejected");
    }

    std::printf("module_divfa: ALL PASSED\n");
    return 0;
}
