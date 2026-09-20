// ---------------------------------------------------------------------------
// module_pw4 — 3/4-argument pointwise extensions (v2.44.0, PHIX_IMPROVEMENTS P-3)
//
// The PoolSectionGPU solver could not express two physics terms in the DSL
// because pw/facePW capped at too few field inputs:
//   • lookup anisotropy: face-flux assembly needs (∂φx, ∂φy, θ) — θ is a
//     per-cell FIELD, not a global constant;
//   • anti-trapping flux: needs (φ, U, ∂tφ, n̂) on the face — four inputs.
//
// Verifies:
//   1. pw(f1,f2,f3,f4): GPU (Equation::computeRHS) and CPU (computeRHSCPU)
//      match the host reference, incl. coeff accumulation.
//   2. facePW / facePWGPU 4-field on x- and y-faces vs host reference.
//   3. Fused fpw3 / fpw4 through fuse() → computeRHS vs host reference.
//   4. Acceptance demo A: per-cell-θ anisotropic face flux (3-field facePW,
//      the k_dphi pattern) — GPU vs CPU paths agree.
//   5. Acceptance demo B: anti-trapping face flux (4-field facePW, the
//      k_updateU pattern) — GPU vs CPU paths agree.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "equation/Term.h"
#include "equation/FusedTerm.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "operators/FaceOps.h"
#include "mesh/Mesh.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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

static void fillPattern(ScalarField& f, double shift) {
    f.fillCurr(0.0);
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            std::sin(0.41 * i + shift) * std::cos(0.29 * j - shift)
            + 0.1 * shift;
}

// ============================================================================
// 1 & 3: cell-centred pw4 and Fused fpw3/fpw4
// ============================================================================

static void testCellPointwise() {
    const int nx = 32, ny = 24;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.5, 0.0,
                                    ny, 0.5, 0.0);
    ScalarField f1(mesh, "f1", 1), f2(mesh, "f2", 1),
                f3(mesh, "f3", 1), f4(mesh, "f4", 1);
    fillPattern(f1, 0.0); fillPattern(f2, 1.3);
    fillPattern(f3, 2.1); fillPattern(f4, -0.7);
    for (ScalarField* f : {&f1, &f2, &f3, &f4}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }

    // the anti-trapping-shaped 4-input functor
    auto fn4 = [] __host__ __device__ (double a, double b, double c, double d) {
        return a * b - 0.5 * (1.0 + b) * c / (fabs(d) + 1.0);
    };
    const double coeff = 1.7;

    ScalarField rhs(mesh, "rhs", 1);
    rhs.allocDevice();

    // GPU path
    Equation eq(f1);
    eq.setRHS(pw(f1, f2, f3, f4, fn4, coeff));
    CUDA_REQUIRE(cudaMemset(rhs.d_curr, 0, rhs.storedBytes()));
    eq.computeRHS(rhs);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    rhs.downloadCurrFromDevice();

    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(f1.index(i, j));
        const double expect =
            coeff * fn4(f1.curr[c], f2.curr[c], f3.curr[c], f4.curr[c]);
        requireClose(rhs.curr[c], expect, 1e-14, "pw4 GPU mismatch");
    }

    // CPU path
    ScalarField rhsC(mesh, "rhsC", 1);
    rhsC.fillCurr(0.0);
    eq.computeRHSCPU(rhsC);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(f1.index(i, j));
        const double expect =
            coeff * fn4(f1.curr[c], f2.curr[c], f3.curr[c], f4.curr[c]);
        requireClose(rhsC.curr[c], expect, 1e-14, "pw4 CPU mismatch");
    }

    // Fused fpw3 + fpw4 in one expression (GPU only — Fused has no CPU path)
    {
        using namespace PhiX::Fused;
        auto fn3 = [] __host__ __device__ (double a, double b, double c) {
            return a * a - b * c;
        };
        Equation eqF(f1);
        eqF.setRHS(fuse(fpw3(f1, f2, f3, fn3) * 2.0 + fpw4(f1, f2, f3, f4, fn4),
                        f1));
        CUDA_REQUIRE(cudaMemset(rhs.d_curr, 0, rhs.storedBytes()));
        eqF.computeRHS(rhs);
        CUDA_REQUIRE(cudaDeviceSynchronize());
        rhs.downloadCurrFromDevice();

        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const std::size_t c = static_cast<std::size_t>(f1.index(i, j));
            const double expect =
                2.0 * fn3(f1.curr[c], f2.curr[c], f3.curr[c])
                + fn4(f1.curr[c], f2.curr[c], f3.curr[c], f4.curr[c]);
            requireClose(rhs.curr[c], expect, 1e-14, "fpw3+fpw4 mismatch");
        }
    }
}

// ============================================================================
// 2, 4 & 5: face-centred pointwise (4-field arity + the two physics demos)
// ============================================================================

// Fill a FaceField (normal axis ax) with a deterministic pattern.
static void fillFace(FaceField& ff, double shift) {
    int lim[3] = { ff.mesh.n[0], ff.mesh.n[1], ff.mesh.n[2] };
    lim[ff.normalAxis] += 1;
    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i)
        ff.data[static_cast<std::size_t>(ff.index(i, j, k))] =
            std::cos(0.53 * i - shift) * std::sin(0.37 * j + 0.5 * shift)
            + 0.05 * shift;
}

static void testFacePointwise() {
    const int nx = 20, ny = 16;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.1, 0.0,
                                    ny, 0.1, 0.0);

    for (int ax : {0, 1}) {
        FaceField a(mesh, ax, "a"), b(mesh, ax, "b"),
                  c(mesh, ax, "c"), d(mesh, ax, "d"),
                  outG(mesh, ax, "outG"), outC(mesh, ax, "outC");
        fillFace(a, 0.0); fillFace(b, 1.1); fillFace(c, -2.3); fillFace(d, 0.6);

        // Demo B functor — anti-trapping flux on a face:
        //   F_at = at * (1 + (1-k)*U_face) * dphi_face * n_component
        // (φ enters the diffusive prefactor D*(1-φ)/2 in the same call)
        const double at = 0.23, kpart = 0.75, D = 3.0;
        auto atFlux = [at, kpart, D] __host__ __device__
                      (double phiF, double uF, double dphiF, double nF) {
            const double q = D * 0.5 * (1.0 - phiF);
            return q * uF + at * (1.0 + (1.0 - kpart) * uF) * dphiF * nF;
        };

        for (FaceField* ff : {&a, &b, &c, &d, &outG}) {
            ff->allocDevice();
            ff->uploadToDevice();
        }
        facePWGPU(outG, a, b, c, d, atFlux);
        CUDA_REQUIRE(cudaDeviceSynchronize());
        outG.downloadFromDevice();

        facePW(outC, a, b, c, d, atFlux);   // CPU path

        int lim[3] = { nx, ny, 1 };
        lim[ax] += 1;
        for (int j = 0; j < lim[1]; ++j)
        for (int i = 0; i < lim[0]; ++i) {
            const std::size_t idx = static_cast<std::size_t>(outG.index(i, j));
            const double expect = atFlux(a.data[idx], b.data[idx],
                                         c.data[idx], d.data[idx]);
            requireClose(outG.data[idx], expect, 1e-14,
                         "facePWGPU(4) ax=" + std::to_string(ax));
            requireClose(outC.data[idx], expect, 1e-14,
                         "facePW(4) ax=" + std::to_string(ax));
        }
    }

    // Demo A — per-cell-θ lookup anisotropy on x-faces (the k_dphi pattern):
    // inputs come from the official cell→face chain (faceGradGPU + interpGPU).
    {
        ScalarField phi(mesh, "phi", 1), gyc(mesh, "gyc", 1),
                    theta(mesh, "theta", 1);
        fillPattern(phi, 0.4);
        fillPattern(theta, -1.9);
        // cell-centred ∂φ/∂y (host, central difference on the padded array —
        // ghost rows are zero-filled which is fine for this arity test)
        gyc.fillCurr(0.0);
        const double invdy = 1.0 / mesh.d[1];
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            gyc.curr[static_cast<std::size_t>(gyc.index(i, j))] =
                0.5 * invdy *
                (phi.curr[static_cast<std::size_t>(phi.index(i, j + 1))] -
                 phi.curr[static_cast<std::size_t>(phi.index(i, j - 1))]);

        for (ScalarField* f : {&phi, &gyc, &theta}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }

        FaceField gpx(mesh, 0, "gpx"), gpy(mesh, 0, "gpy"),
                  thf(mesh, 0, "thf"), jxG(mesh, 0, "jxG"), jxC(mesh, 0, "jxC");
        for (FaceField* ff : {&gpx, &gpy, &thf, &jxG}) ff->allocDevice();

        faceGradGPU(phi, 0, gpx);    // ∂φ/∂x on x-faces
        interpGPU(gyc, 0, gpy);      // ∂φ/∂y interpolated to x-faces
        interpGPU(theta, 0, thf);    // per-cell θ interpolated to x-faces

        const double eps4 = 0.05, W0sq = 1.7;
        auto anisoFlux = [eps4, W0sq] __host__ __device__
                         (double px, double py, double th) {
            const double psi = atan2(py, px + 1e-30) - th;
            const double a   = 1.0 + eps4 * cos(4.0 * psi);
            const double s4  = 4.0 * eps4 * sin(4.0 * psi);
            return W0sq * a * (a * px + s4 * py);
        };
        facePWGPU(jxG, gpx, gpy, thf, anisoFlux);   // 3-field, θ is a FIELD
        CUDA_REQUIRE(cudaDeviceSynchronize());

        for (FaceField* ff : {&gpx, &gpy, &thf, &jxG}) ff->downloadFromDevice();
        facePW(jxC, gpx, gpy, thf, anisoFlux);      // CPU path on same inputs

        bool sawNonzero = false;
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            const std::size_t idx = static_cast<std::size_t>(jxG.index(i, j));
            requireClose(jxG.data[idx], jxC.data[idx], 1e-12,
                         "aniso flux GPU vs CPU");
            if (std::fabs(jxG.data[idx]) > 1e-3) sawNonzero = true;
        }
        require(sawNonzero, "aniso flux demo produced all-zero fluxes");
    }
}

int main() {
    testCellPointwise();
    testFacePointwise();
    std::printf("module_pw4: ALL PASSED\n");
    return 0;
}
