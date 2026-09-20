#include "operators/FaceOps.h"
#include "core/Check.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// ---------------------------------------------------------------------------
// Inline helpers (host + device)
// ---------------------------------------------------------------------------

// Face flat index matching FieldLayout::index() for FACE_X/Y/Z.
__host__ __device__ inline
int face_idx(int i, int j, int k, int ax, int sx_f, int sy_f, int g)
{
    int si = (ax == 0) ? i : (i + g);
    int sj = (ax == 1) ? j : (j + g);
    int sk = (ax == 2) ? k : (k + g);
    return si + sx_f * (sj + sy_f * sk);
}

// Cell flat index (always uses +ghost in all axes).
__host__ __device__ inline
int cell_idx(int i, int j, int k, int sx_c, int sy_c, int g)
{
    return (i + g) + sx_c * ((j + g) + sy_c * (k + g));
}

// ===========================================================================
// CUDA kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// kernel_interp
//
// Threads are mapped over the full face set (lim[0]*lim[1]*lim[2]) where
//   lim[ax] = n[ax] + 1, lim[other] = n[other].
// Interior faces: arithmetic mean; boundary faces: nearest-cell value.
// ---------------------------------------------------------------------------
__global__ void kernel_interp(
        Real*       f_data,
        const Real* c_data,
        int lim0, int lim1, int lim2,
        int ax,
        int n_ax,               // = mesh.n[ax]
        int sx_c, int sy_c,
        int sx_f, int sy_f,
        int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = lim0 * lim1 * lim2;
    if (tid >= total) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int fidx = face_idx(i, j, k, ax, sx_f, sy_f, g);

    // Normal index
    int i_n = (ax == 0) ? i : (ax == 1) ? j : k;

    // Right cell: at (i, j, k) which are within [0, n[ax]] for normal axis.
    // For i_n == n_ax, the "right" cell is out of physical range — use left only.
    int ridx = cell_idx(i, j, k, sx_c, sy_c, g);  // valid for i_n < n_ax

    if (i_n == 0) {
        f_data[fidx] = c_data[ridx];
    } else if (i_n == n_ax) {
        // left cell: decrement normal coordinate
        int li = i - (ax == 0 ? 1 : 0);
        int lj = j - (ax == 1 ? 1 : 0);
        int lk = k - (ax == 2 ? 1 : 0);
        f_data[fidx] = c_data[cell_idx(li, lj, lk, sx_c, sy_c, g)];
    } else {
        int li = i - (ax == 0 ? 1 : 0);
        int lj = j - (ax == 1 ? 1 : 0);
        int lk = k - (ax == 2 ? 1 : 0);
        int lidx = cell_idx(li, lj, lk, sx_c, sy_c, g);
        f_data[fidx] = 0.5 * (c_data[lidx] + c_data[ridx]);
    }
}

// ---------------------------------------------------------------------------
// kernel_face_grad
//
// g_face[i,j,k] = (cell[i,j,k] - cell[i-1,j,k]) / dx  (normal axis)
// Boundary faces (i_n=0) use ghost cell (valid if BCs have been applied).
// ---------------------------------------------------------------------------
__global__ void kernel_face_grad(
        Real*       f_data,
        const Real* c_data,
        int lim0, int lim1, int lim2,
        int ax,
        int sx_c, int sy_c,
        int sx_f, int sy_f,
        int g,
        double inv_d)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = lim0 * lim1 * lim2;
    if (tid >= total) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int fidx = face_idx(i, j, k, ax, sx_f, sy_f, g);
    int ridx  = cell_idx(i, j, k, sx_c, sy_c, g);

    int li = i - (ax == 0 ? 1 : 0);
    int lj = j - (ax == 1 ? 1 : 0);
    int lk = k - (ax == 2 ? 1 : 0);
    // left index: when i_n=0, li=-1, li+g = g-1 = 0 for g=1 → ghost cell
    int lidx = cell_idx(li, lj, lk, sx_c, sy_c, g);

    f_data[fidx] = (c_data[ridx] - c_data[lidx]) * inv_d;
}

// ---------------------------------------------------------------------------
// kernel_div_face
//
// rhs[i,j,k] += coeff * divergence of face fluxes (conservative FV)
// ---------------------------------------------------------------------------
__global__ void kernel_div_face(
        Real*       rhs,
        const Real* fx,
        const Real* fy,
        const Real* fz,
        double coeff,
        int nx, int ny, int nz,
        int sx_c, int sy_c,
        int sx_fx, int sy_fx,
        int sx_fy, int sy_fy,
        int sx_fz, int sy_fz,
        int g,
        double inv_dx, double inv_dy, double inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int cidx = cell_idx(i, j, k, sx_c, sy_c, g);

    double val = 0.0;

    if (fx) {
        // FACE_X: si = i (no ghost), sj = j+g, sk = k+g
        int ip1 = (i+1) + sx_fx * ((j+g) + sy_fx * (k+g));
        int ic  =  i    + sx_fx * ((j+g) + sy_fx * (k+g));
        val += (fx[ip1] - fx[ic]) * inv_dx;
    }
    if (fy) {
        // FACE_Y: si = i+g, sj = j (no ghost), sk = k+g
        int jp1 = (i+g) + sx_fy * ((j+1) + sy_fy * (k+g));
        int jc  = (i+g) + sx_fy * ( j    + sy_fy * (k+g));
        val += (fy[jp1] - fy[jc]) * inv_dy;
    }
    if (fz) {
        // FACE_Z: si = i+g, sj = j+g, sk = k (no ghost)
        int kp1 = (i+g) + sx_fz * ((j+g) + sy_fz * (k+1));
        int kc  = (i+g) + sx_fz * ((j+g) + sy_fz *  k   );
        val += (fz[kp1] - fz[kc]) * inv_dz;
    }

    rhs[cidx] += coeff * val;
}

} // anonymous namespace

// ===========================================================================
// CPU implementations
// ===========================================================================

void interp(const ScalarField& cell, int ax, FaceField& face)
{
    if (ax != face.normalAxis)
        throw std::invalid_argument("interp: axis mismatch with FaceField.normalAxis");

    const int g  = cell.ghost;
    const int sx = cell.storedDims[0], sy = cell.storedDims[1];
    const int n  = cell.mesh.n[ax];

    // Loop bounds: normal axis → n+1 faces; others → n physical cells
    int lim[3] = { cell.mesh.n[0], cell.mesh.n[1], cell.mesh.n[2] };
    lim[ax] += 1;

    const auto& cdat = cell.curr;
    auto&       fdat = face.data;

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int fidx = face.index(i, j, k);
        int i_n  = (ax == 0) ? i : (ax == 1) ? j : k;
        int ridx = cell_idx(i, j, k, sx, sy, g);

        if (i_n == 0) {
            fdat[fidx] = cdat[ridx];
        } else if (i_n == n) {
            int li = i - (ax == 0 ? 1 : 0);
            int lj = j - (ax == 1 ? 1 : 0);
            int lk = k - (ax == 2 ? 1 : 0);
            fdat[fidx] = cdat[cell_idx(li, lj, lk, sx, sy, g)];
        } else {
            int li = i - (ax == 0 ? 1 : 0);
            int lj = j - (ax == 1 ? 1 : 0);
            int lk = k - (ax == 2 ? 1 : 0);
            fdat[fidx] = 0.5 * (cdat[cell_idx(li, lj, lk, sx, sy, g)] + cdat[ridx]);
        }
    }
}

void faceGrad(const ScalarField& cell, int ax, FaceField& face)
{
    if (ax != face.normalAxis)
        throw std::invalid_argument("faceGrad: axis mismatch with FaceField.normalAxis");

    const int g      = cell.ghost;
    const int sx     = cell.storedDims[0], sy = cell.storedDims[1];
    const double inv = 1.0 / cell.mesh.d[ax];

    int lim[3] = { cell.mesh.n[0], cell.mesh.n[1], cell.mesh.n[2] };
    lim[ax] += 1;

    const auto& cdat = cell.curr;
    auto&       fdat = face.data;

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int fidx = face.index(i, j, k);
        int ridx = cell_idx(i, j, k, sx, sy, g);
        int li = i - (ax == 0 ? 1 : 0);
        int lj = j - (ax == 1 ? 1 : 0);
        int lk = k - (ax == 2 ? 1 : 0);
        // When i_n=0: li = -1 (ghost cell), li+g = 0 for g=1 → valid.
        fdat[fidx] = (cdat[ridx] - cdat[cell_idx(li, lj, lk, sx, sy, g)]) * inv;
    }
}

// ===========================================================================
// GPU implementations
// ===========================================================================

void interpGPU(const ScalarField& cell, int ax, FaceField& face)
{
    if (ax != face.normalAxis)
        throw std::invalid_argument("interpGPU: axis mismatch");
    if (!cell.d_curr)
        throw std::runtime_error("interpGPU: cell field not on device");
    if (!face.d_data)
        throw std::runtime_error("interpGPU: face field not allocated on device");

    int lim[3] = { cell.mesh.n[0], cell.mesh.n[1], cell.mesh.n[2] };
    lim[ax] += 1;
    int total = lim[0] * lim[1] * lim[2];

    kernel_interp<<<(total+255)/256, 256>>>(
        face.d_data, cell.d_curr,
        lim[0], lim[1], lim[2], ax, cell.mesh.n[ax],
        cell.storedDims[0], cell.storedDims[1],
        face.storedDims[0], face.storedDims[1],
        cell.ghost);
    CUDA_CHECK(cudaGetLastError());
}

void faceGradGPU(const ScalarField& cell, int ax, FaceField& face)
{
    if (ax != face.normalAxis)
        throw std::invalid_argument("faceGradGPU: axis mismatch");
    if (!cell.d_curr)
        throw std::runtime_error("faceGradGPU: cell field not on device");
    if (!face.d_data)
        throw std::runtime_error("faceGradGPU: face field not allocated on device");

    int lim[3] = { cell.mesh.n[0], cell.mesh.n[1], cell.mesh.n[2] };
    lim[ax] += 1;
    int total = lim[0] * lim[1] * lim[2];

    kernel_face_grad<<<(total+255)/256, 256>>>(
        face.d_data, cell.d_curr,
        lim[0], lim[1], lim[2], ax,
        cell.storedDims[0], cell.storedDims[1],
        face.storedDims[0], face.storedDims[1],
        cell.ghost, 1.0 / cell.mesh.d[ax]);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// divFace — returns a Term
// ===========================================================================

Term divFace(const FaceField* fx,
             const FaceField* fy,
             const FaceField* fz,
             double coeff)
{
    const Mesh* mp = nullptr;
    int ghost = 1;
    if (fx) { mp = &fx->mesh; ghost = fx->ghost; }
    else if (fy) { mp = &fy->mesh; ghost = fy->ghost; }
    else if (fz) { mp = &fz->mesh; ghost = fz->ghost; }
    else throw std::invalid_argument("divFace: all flux pointers are null");

    // Every provided flux must live on the same mesh/ghost as the reference
    // (the kernel takes geometry from the first non-null one), and sit in the
    // slot matching its normal axis — a swapped-argument bug otherwise
    // produces a silently wrong divergence.
    check::checkCartesian(*mp, "divFace");
    const FaceField* fluxes[3] = { fx, fy, fz };
    for (int ax = 0; ax < 3; ++ax) {
        const FaceField* ff = fluxes[ax];
        if (!ff) continue;
        if (ff->normalAxis != ax)
            throw ValidationError("divFace",
                "flux argument " + std::to_string(ax) + " has normalAxis = "
                + std::to_string(ff->normalAxis),
                "pass the x/y/z FaceFields in order: divFace(&fx, &fy, &fz)");
        if (!check::sameMeshGeometry(ff->mesh, *mp) || ff->ghost != ghost)
            throw ValidationError("divFace",
                "flux fields do not share the same mesh geometry and ghost "
                "width (axis " + std::to_string(ax) + " differs)",
                "build all FaceFields from the same Mesh with the same ghost");
    }

    const Mesh& mesh = *mp;
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
    const int sx_c = nx + 2*ghost,  sy_c = ny + 2*ghost;

    const int sx_fx = fx ? fx->storedDims[0] : 0, sy_fx = fx ? fx->storedDims[1] : 0;
    const int sx_fy = fy ? fy->storedDims[0] : 0, sy_fy = fy ? fy->storedDims[1] : 0;
    const int sx_fz = fz ? fz->storedDims[0] : 0, sy_fz = fz ? fz->storedDims[1] : 0;

    const double inv_dx = 1.0 / mesh.d[0];
    const double inv_dy = (mesh.dim >= 2) ? 1.0 / mesh.d[1] : 0.0;
    const double inv_dz = (mesh.dim >= 3) ? 1.0 / mesh.d[2] : 0.0;

    Term t;
    t.type         = TermType::COMPOSITE;
    t.coeff        = coeff;
    t.field        = nullptr;
    t.ghostRequired = 0;
    t.info.complete = true;
    for (auto *face : fluxes)
        t.info.read(face);
    t.rhsGhost     = ghost;   // kernel indexes the rhs with sx_c = nx + 2*ghost

    t.gpu_launcher = [fx, fy, fz, nx, ny, nz,
                      sx_c, sy_c,
                      sx_fx, sy_fx, sx_fy, sy_fy, sx_fz, sy_fz,
                      ghost, inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool)
    {
        if (fx && !fx->d_data)
            throw std::runtime_error("divFace GPU: flux_x not on device");
        if (fy && !fy->d_data)
            throw std::runtime_error("divFace GPU: flux_y not on device");
        if (fz && !fz->d_data)
            throw std::runtime_error("divFace GPU: flux_z not on device");

        const Real* dfx = fx ? fx->d_data : nullptr;
        const Real* dfy = fy ? fy->d_data : nullptr;
        const Real* dfz = fz ? fz->d_data : nullptr;

        int total = nx * ny * nz;
        kernel_div_face<<<(total+255)/256, 256, 0, pool.stream>>>(
            d_rhs, dfx, dfy, dfz, c,
            nx, ny, nz, sx_c, sy_c,
            sx_fx, sy_fx, sx_fy, sy_fy, sx_fz, sy_fz,
            ghost, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("divFace GPU kernel: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [fx, fy, fz, nx, ny, nz,
                    sx_c, sy_c,
                    sx_fx, sy_fx, sx_fy, sy_fy, sx_fz, sy_fz,
                    ghost, inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&)
    {
        const Real* cfx = fx ? fx->data.data() : nullptr;
        const Real* cfy = fy ? fy->data.data() : nullptr;
        const Real* cfz = fz ? fz->data.data() : nullptr;

        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int cidx = cell_idx(i, j, k, sx_c, sy_c, ghost);
            double val = 0.0;

            if (cfx) {
                int ip1 = (i+1) + sx_fx * ((j+ghost) + sy_fx * (k+ghost));
                int ic  =  i    + sx_fx * ((j+ghost) + sy_fx * (k+ghost));
                val += (cfx[ip1] - cfx[ic]) * inv_dx;
            }
            if (cfy) {
                int jp1 = (i+ghost) + sx_fy * ((j+1) + sy_fy * (k+ghost));
                int jc  = (i+ghost) + sx_fy * ( j    + sy_fy * (k+ghost));
                val += (cfy[jp1] - cfy[jc]) * inv_dy;
            }
            if (cfz) {
                int kp1 = (i+ghost) + sx_fz * ((j+ghost) + sy_fz * (k+1));
                int kc  = (i+ghost) + sx_fz * ((j+ghost) + sy_fz *  k   );
                val += (cfz[kp1] - cfz[kc]) * inv_dz;
            }
            rhs[cidx] += c * val;
        }
    };

    return t;
}

} // namespace PhiX
