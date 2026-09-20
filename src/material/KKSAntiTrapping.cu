#include "material/KKSAntiTrapping.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

void KKSAntiTrappingParams::validate() const {
    if (W <= 0.0)
        throw std::invalid_argument("KKSAntiTrappingParams: W must be > 0");
    if (a <= 0.0)
        throw std::invalid_argument("KKSAntiTrappingParams: a must be > 0");
    if (gradEps < 0.0)
        throw std::invalid_argument("KKSAntiTrappingParams: gradEps must be >= 0");
}

namespace {

// Index helpers (mirror src/operators/FaceOps.cu conventions)
__host__ __device__ inline
int cell_idx(int i, int j, int k, int sx_c, int sy_c, int g) {
    return (i + g) + sx_c * ((j + g) + sy_c * (k + g));
}

__host__ __device__ inline
int face_idx(int i, int j, int k, int ax, int sx_f, int sy_f, int g) {
    int si = (ax == 0) ? i : (i + g);
    int sj = (ax == 1) ? j : (j + g);
    int sk = (ax == 2) ? k : (k + g);
    return si + sx_f * (sj + sy_f * sk);
}

// ---------------------------------------------------------------------------
// Face value of j_at along `ax` between cells L and R (shared CPU/GPU body).
// Returns 0 away from the interface (|∇φ| <= gradEps).
// ---------------------------------------------------------------------------
__host__ __device__ inline
Real atFaceValue(const Real* c, const Real* phi, const Real* dpdt,
                 KKSView v,
                 int L, int R, int ax, int dim,
                 int sx_c, int sy_c,
                 Real inv_dx, Real inv_dy, Real inv_dz,
                 Real aW, Real gradEps)
{
    const Real inv_dn = (ax == 0) ? inv_dx : (ax == 1) ? inv_dy : inv_dz;
    const Real gn = (phi[R] - phi[L]) * inv_dn;

    // |∇φ|² at the face: normal component + averaged transverse central diffs
    Real g2 = gn * gn;
    for (int b = 0; b < dim; ++b) {
        if (b == ax) continue;
        const int  sb     = (b == 0) ? 1 : (b == 1) ? sx_c : sx_c * sy_c;
        const Real inv_db = (b == 0) ? inv_dx : (b == 1) ? inv_dy : inv_dz;
        const Real gt = Real(0.25) * inv_db
                      * (phi[L + sb] - phi[L - sb] + phi[R + sb] - phi[R - sb]);
        g2 += gt * gt;
    }

    const Real gmag = sqrt(g2);
    if (gmag <= gradEps) return Real(0);

    Real phi_f = Real(0.5) * (phi[L] + phi[R]);
    phi_f = (phi_f < Real(0)) ? Real(0) : (phi_f > Real(1)) ? Real(1) : phi_f;
    const Real c_f = Real(0.5) * (c[L] + c[R]);

    Real cs, cl, mu;
    v.partition(c_f, kks::h(phi_f), cs, cl, mu);

    const Real dp_f = Real(0.5) * (dpdt[L] + dpdt[R]);

    // PhiX's divFace accumulates rhs += +∇·F, i.e. the face fields hold the
    // NEGATIVE of the physical current (F = M∇μ = −J_diff).  The physical
    // anti-trapping current  J_at = −aW(c_l−c_s)(∂φ/∂t)∇φ/|∇φ|  therefore
    // enters the face field as −J_at = +aW(c_l−c_s)(∂φ/∂t)∇φ/|∇φ|.
    return aW * (cl - cs) * dp_f * gn / gmag;
}

__global__ void kernel_kks_at(
        Real* f_data,
        const Real* c, const Real* phi, const Real* dpdt,
        KKSView v,
        int lim0, int lim1, int lim2, int ax, int dim,
        int sx_c, int sy_c,
        int sx_f, int sy_f,
        int g,
        Real inv_dx, Real inv_dy, Real inv_dz,
        Real aW, Real gradEps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lim0 * lim1 * lim2) return;

    const int i = tid % lim0;
    const int j = (tid / lim0) % lim1;
    const int k = tid / (lim0 * lim1);

    const int li = i - (ax == 0 ? 1 : 0);
    const int lj = j - (ax == 1 ? 1 : 0);
    const int lk = k - (ax == 2 ? 1 : 0);

    const int R = cell_idx(i, j, k, sx_c, sy_c, g);
    const int L = cell_idx(li, lj, lk, sx_c, sy_c, g);

    f_data[face_idx(i, j, k, ax, sx_f, sy_f, g)]
        += atFaceValue(c, phi, dpdt, v, L, R, ax, dim,
                       sx_c, sy_c, inv_dx, inv_dy, inv_dz, aW, gradEps);
}

void checkInputs(const KKSAntiTrappingParams& p,
                 const ScalarField& c, const ScalarField& phi,
                 const ScalarField& dphidt,
                 FaceField* faces[3], const char* fn)
{
    p.validate();
    if (phi.storedSize != c.storedSize || dphidt.storedSize != c.storedSize
        || phi.ghost != c.ghost || dphidt.ghost != c.ghost)
        throw std::invalid_argument(
            std::string(fn) + ": c/phi/dphidt layouts differ");
    if (c.ghost < 1)
        throw std::invalid_argument(
            std::string(fn) + ": fields need ghost >= 1");
    for (int ax = 0; ax < 3; ++ax) {
        const bool active = ax < c.mesh.dim;
        if (active && !faces[ax])
            throw std::invalid_argument(
                std::string(fn) + ": missing face field for active axis "
                + std::to_string(ax));
        if (!active && faces[ax])
            throw std::invalid_argument(
                std::string(fn) + ": face field given for inactive axis "
                + std::to_string(ax));
        if (faces[ax] && faces[ax]->normalAxis != ax)
            throw std::invalid_argument(
                std::string(fn) + ": face field normalAxis mismatch on axis "
                + std::to_string(ax));
    }
}

} // namespace

void kksAddAntiTrappingGPU(const KKSParabolic& model,
                           const KKSAntiTrappingParams& p,
                           const ScalarField& c,
                           const ScalarField& phi,
                           const ScalarField& dphidt,
                           FaceField* jx, FaceField* jy, FaceField* jz)
{
    FaceField* faces[3] = {jx, jy, jz};
    checkInputs(p, c, phi, dphidt, faces, "kksAddAntiTrappingGPU");
    if (!c.d_curr || !phi.d_curr || !dphidt.d_curr)
        throw std::runtime_error(
            "kksAddAntiTrappingGPU: cell fields need device allocation");

    const int  dim    = c.mesh.dim;
    const Real inv_dx = static_cast<Real>(1.0 / c.mesh.d[0]);
    const Real inv_dy = (dim >= 2) ? static_cast<Real>(1.0 / c.mesh.d[1]) : Real(0);
    const Real inv_dz = (dim >= 3) ? static_cast<Real>(1.0 / c.mesh.d[2]) : Real(0);
    const Real aW      = static_cast<Real>(p.a * p.W);
    const Real gradEps = static_cast<Real>(p.gradEps);

    for (int ax = 0; ax < dim; ++ax) {
        FaceField& f = *faces[ax];
        if (!f.d_data)
            throw std::runtime_error(
                "kksAddAntiTrappingGPU: face field not on device");
        int lim[3] = {c.mesh.n[0], c.mesh.n[1], c.mesh.n[2]};
        lim[ax] += 1;
        const int total = lim[0] * lim[1] * lim[2];

        kernel_kks_at<<<(total + 255) / 256, 256>>>(
            f.d_data, c.d_curr, phi.d_curr, dphidt.d_curr, model.view(),
            lim[0], lim[1], lim[2], ax, dim,
            c.storedDims[0], c.storedDims[1],
            f.storedDims[0], f.storedDims[1],
            c.ghost, inv_dx, inv_dy, inv_dz, aW, gradEps);
        CUDA_CHECK(cudaGetLastError());
    }
}

void kksAddAntiTrappingCPU(const KKSParabolic& model,
                           const KKSAntiTrappingParams& p,
                           const ScalarField& c,
                           const ScalarField& phi,
                           const ScalarField& dphidt,
                           FaceField* jx, FaceField* jy, FaceField* jz)
{
    FaceField* faces[3] = {jx, jy, jz};
    checkInputs(p, c, phi, dphidt, faces, "kksAddAntiTrappingCPU");

    const KKSView v   = model.view();
    const int  dim    = c.mesh.dim;
    const Real inv_dx = static_cast<Real>(1.0 / c.mesh.d[0]);
    const Real inv_dy = (dim >= 2) ? static_cast<Real>(1.0 / c.mesh.d[1]) : Real(0);
    const Real inv_dz = (dim >= 3) ? static_cast<Real>(1.0 / c.mesh.d[2]) : Real(0);
    const Real aW      = static_cast<Real>(p.a * p.W);
    const Real gradEps = static_cast<Real>(p.gradEps);
    const int  g = c.ghost;

    for (int ax = 0; ax < dim; ++ax) {
        FaceField& f = *faces[ax];
        int lim[3] = {c.mesh.n[0], c.mesh.n[1], c.mesh.n[2]};
        lim[ax] += 1;
        for (int k = 0; k < lim[2]; ++k)
        for (int j = 0; j < lim[1]; ++j)
        for (int i = 0; i < lim[0]; ++i) {
            const int li = i - (ax == 0 ? 1 : 0);
            const int lj = j - (ax == 1 ? 1 : 0);
            const int lk = k - (ax == 2 ? 1 : 0);
            const int R = cell_idx(i, j, k,
                                   c.storedDims[0], c.storedDims[1], g);
            const int L = cell_idx(li, lj, lk,
                                   c.storedDims[0], c.storedDims[1], g);
            f.data[static_cast<std::size_t>(
                face_idx(i, j, k, ax, f.storedDims[0], f.storedDims[1], g))]
                += atFaceValue(c.curr.data(), phi.curr.data(),
                               dphidt.curr.data(), v, L, R, ax, dim,
                               c.storedDims[0], c.storedDims[1],
                               inv_dx, inv_dy, inv_dz, aW, gradEps);
        }
    }
}

} // namespace PhiX
