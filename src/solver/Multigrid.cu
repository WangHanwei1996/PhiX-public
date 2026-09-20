#include "solver/Multigrid.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

inline int blocks(int n) { return (n + 255) / 256; }

// Neighbour index helpers: periodic wraps, no-flux clamps (the clamped value
// is multiplied by a zeroed boundary-face coefficient, so it never matters).
__device__ inline int nbr(int i, int n, int step, int periodic)
{
    int j = i + step;
    if (periodic) {
        if (j < 0) j += n;
        else if (j >= n) j -= n;
    } else {
        if (j < 0) j = 0;
        else if (j >= n) j = n - 1;
    }
    return j;
}

// A z at cell (i,j):  alpha·z − sigma·[ax_{i+1}(z_E−z)−ax_i(z−z_W)]/dx²
//                              − sigma·[ay_{j+1}(z_N−z)−ay_j(z−z_S)]/dy²
__device__ inline Real applyA(const Real* z, const double* ax,
                              const double* ay, int i, int j, int nx, int ny,
                              double inv_dx2, double inv_dy2,
                              double alpha, double sigma, int px, int py)
{
    const int    c  = i + nx * j;
    const double zc = static_cast<double>(z[c]);
    const double zW = z[nbr(i, nx, -1, px) + nx * j];
    const double zE = z[nbr(i, nx, +1, px) + nx * j];
    const double zS = z[i + nx * nbr(j, ny, -1, py)];
    const double zN = z[i + nx * nbr(j, ny, +1, py)];
    const double axW = ax[i     + (nx + 1) * j];
    const double axE = ax[i + 1 + (nx + 1) * j];
    const double ayS = ay[i + nx * j];
    const double ayN = ay[i + nx * (j + 1)];
    const double lap = (axE * (zE - zc) - axW * (zc - zW)) * inv_dx2
                     + (ayN * (zN - zc) - ayS * (zc - zS)) * inv_dy2;
    return static_cast<Real>(alpha * zc - sigma * lap);
}

// t = r − A z    (Jacobi residual snapshot — reads a CONSISTENT z)
__global__ void k_mg_residual(Real* t, const Real* r, const Real* z,
                              const double* ax, const double* ay,
                              int nx, int ny, double inv_dx2, double inv_dy2,
                              const double* alpha, const double* sigma,
                              int px, int py)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    const int i = id % nx, j = id / nx;
    t[id] = r[id] - applyA(z, ax, ay, i, j, nx, ny, inv_dx2, inv_dy2,
                           *alpha, *sigma, px, py);
}

// z += omega · t / diag(A)     (pointwise — deterministic Jacobi second half)
__global__ void k_mg_jacobi(Real* z, const Real* t,
                            const double* ax, const double* ay,
                            int nx, int ny, double inv_dx2, double inv_dy2,
                            const double* alpha, const double* sigma,
                            double omega)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    const int i = id % nx, j = id / nx;
    const double diag = *alpha + *sigma
        * ((ax[i + (nx + 1) * j] + ax[i + 1 + (nx + 1) * j]) * inv_dx2
         + (ay[i + nx * j]       + ay[i + nx * (j + 1)])     * inv_dy2);
    z[id] += static_cast<Real>(omega * static_cast<double>(t[id]) / diag);
}

// Coarse r = (P^T/4) t_fine — 16-point gather, weights ([1,3,3,1]⊗[1,3,3,1])/64.
__global__ void k_mg_restrict(Real* rc, const Real* tf,
                              int nxc, int nyc, int nxf, int nyf,
                              int px, int py)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nxc * nyc) return;
    const int I = id % nxc, J = id / nxc;
    const double w1[4] = {1.0, 3.0, 3.0, 1.0};
    double acc = 0.0;
    for (int b = 0; b < 4; ++b) {
        const int jf = nbr(2 * J - 1 + b, nyf, 0, py);
        for (int a = 0; a < 4; ++a) {
            const int ifx = nbr(2 * I - 1 + a, nxf, 0, px);
            acc += w1[a] * w1[b] * static_cast<double>(tf[ifx + nxf * jf]);
        }
    }
    rc[id] = static_cast<Real>(acc / 64.0);
}

// Fine z += P(z_coarse) — bilinear, weights {9,3,3,1}/16.
__global__ void k_mg_prolong_add(Real* zf, const Real* zc,
                                 int nxf, int nyf, int nxc, int nyc,
                                 int px, int py)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nxf * nyf) return;
    const int i = id % nxf, j = id / nxf;
    const int I = i / 2, J = j / 2;
    const int In = nbr(I, nxc, (i & 1) ? +1 : -1, px);
    const int Jn = nbr(J, nyc, (j & 1) ? +1 : -1, py);
    const double v = 9.0 * static_cast<double>(zc[I  + nxc * J ])
                   + 3.0 * static_cast<double>(zc[In + nxc * J ])
                   + 3.0 * static_cast<double>(zc[I  + nxc * Jn])
                   + 1.0 * static_cast<double>(zc[In + nxc * Jn]);
    zf[id] += static_cast<Real>(v / 16.0);
}

// Coarse x-faces: average the two overlapping fine faces.
__global__ void k_mg_coarsen_ax(double* axc, const double* axf,
                                int nxc, int nyc, int nxf)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (nxc + 1) * nyc) return;
    const int I = id % (nxc + 1), J = id / (nxc + 1);
    axc[id] = 0.5 * (axf[2 * I + (nxf + 1) * (2 * J)]
                   + axf[2 * I + (nxf + 1) * (2 * J + 1)]);
}

__global__ void k_mg_coarsen_ay(double* ayc, const double* ayf,
                                int nxc, int nyc, int nxf)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nxc * (nyc + 1)) return;
    const int I = id % nxc, J = id / nxc;
    ayc[id] = 0.5 * (ayf[2 * I     + nxf * (2 * J)]
                   + ayf[2 * I + 1 + nxf * (2 * J)]);
}

__global__ void k_mg_fill(double* a, double v, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

// Zero the boundary faces (no-flux absorption).
__global__ void k_mg_zero_bx(double* ax, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    ax[0  + (nx + 1) * j] = 0.0;
    ax[nx + (nx + 1) * j] = 0.0;
}

__global__ void k_mg_zero_by(double* ay, int nx, int ny)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    ay[i + nx * 0]  = 0.0;
    ay[i + nx * ny] = 0.0;
}

// Pack physical cells of a halo'd ScalarField into a dense level-0 buffer
// (and the reverse for the solution).  planeOff = sx·sy·g — a 2D field
// stores 1+2g z-planes with the physical plane in the middle (see CLAUDE.md).
__global__ void k_mg_pack(Real* dst, const Real* src, int nx, int ny,
                          int sx, int g, int planeOff)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    const int i = id % nx, j = id / nx;
    dst[id] = src[planeOff + (i + g) + sx * (j + g)];
}

__global__ void k_mg_unpack(Real* dst, const Real* src, int nx, int ny,
                            int sx, int g, int planeOff)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;
    const int i = id % nx, j = id / nx;
    dst[planeOff + (i + g) + sx * (j + g)] = src[id];
}

} // namespace

// ===========================================================================
// MultigridPreconditioner
// ===========================================================================

MultigridPreconditioner::MultigridPreconditioner(const Mesh& mesh, BCKind bcX,
                                                 BCKind bcY, int nPre,
                                                 int nPost, double omega,
                                                 int coarseSweeps)
    : bcX_(bcX), bcY_(bcY)
    , nPre_(nPre), nPost_(nPost), coarseSweeps_(coarseSweeps)
    , omega_(omega)
{
    if (mesh.dim != 2)
        throw std::invalid_argument(
            "MultigridPreconditioner: 2D only (dim == 2); 3D is a planned "
            "extension");
    if (nPre < 1 || nPost < 1 || coarseSweeps < 1)
        throw std::invalid_argument(
            "MultigridPreconditioner: sweep counts must be >= 1");
    if (omega <= 0.0 || omega > 1.0)
        throw std::invalid_argument(
            "MultigridPreconditioner: omega must be in (0, 1]");

    int nx = mesh.n[0], ny = mesh.n[1];
    double dx = mesh.d[0], dy = mesh.d[1];
    while (true) {
        Level L;
        L.nx = nx;  L.ny = ny;
        L.inv_dx2 = 1.0 / (dx * dx);
        L.inv_dy2 = 1.0 / (dy * dy);
        const int nc = nx * ny;
        CUDA_CHECK(cudaMalloc(&L.ax, (nx + 1) * ny * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&L.ay, nx * (ny + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&L.z, nc * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&L.r, nc * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&L.t, nc * sizeof(Real)));
        CUDA_CHECK(cudaMemset(L.z, 0, nc * sizeof(Real)));
        CUDA_CHECK(cudaMemset(L.r, 0, nc * sizeof(Real)));
        CUDA_CHECK(cudaMemset(L.t, 0, nc * sizeof(Real)));
        lv_.push_back(L);
        if (nx % 2 || ny % 2 || nx <= 4 || ny <= 4) break;
        nx /= 2;  ny /= 2;  dx *= 2.0;  dy *= 2.0;
    }
}

MultigridPreconditioner::~MultigridPreconditioner() {
    for (auto& L : lv_) {           // best-effort; no throw in dtor
        cudaFree(L.ax);  cudaFree(L.ay);
        cudaFree(L.z);   cudaFree(L.r);   cudaFree(L.t);
    }
}

void MultigridPreconditioner::buildCoarseFaces(cudaStream_t stream) {
    for (std::size_t l = 1; l < lv_.size(); ++l) {
        Level& C = lv_[l];
        Level& F = lv_[l - 1];
        k_mg_coarsen_ax<<<blocks((C.nx + 1) * C.ny), 256, 0, stream>>>(
            C.ax, F.ax, C.nx, C.ny, F.nx);
        k_mg_coarsen_ay<<<blocks(C.nx * (C.ny + 1)), 256, 0, stream>>>(
            C.ay, F.ay, C.nx, C.ny, F.nx);
        CUDA_CHECK(cudaGetLastError());
    }
    // No-flux boundary faces stay zero on every level (coarsening averages
    // two zero fine faces into a zero coarse face) — enforce anyway.
    if (bcX_ == BCKind::NoFlux)
        for (auto& L : lv_) {
            k_mg_zero_bx<<<blocks(L.ny), 256, 0, stream>>>(L.ax, L.nx, L.ny);
            CUDA_CHECK(cudaGetLastError());
        }
    if (bcY_ == BCKind::NoFlux)
        for (auto& L : lv_) {
            k_mg_zero_by<<<blocks(L.nx), 256, 0, stream>>>(L.ay, L.nx, L.ny);
            CUDA_CHECK(cudaGetLastError());
        }
    facesReady_ = true;
}

void MultigridPreconditioner::setup(double a) {
    Level& F = lv_[0];
    k_mg_fill<<<blocks((F.nx + 1) * F.ny), 256>>>(F.ax, a,
                                                  (F.nx + 1) * F.ny);
    k_mg_fill<<<blocks(F.nx * (F.ny + 1)), 256>>>(F.ay, a,
                                                  F.nx * (F.ny + 1));
    CUDA_CHECK(cudaGetLastError());
    buildCoarseFaces();
}

void MultigridPreconditioner::setupFaces(const double* d_ax,
                                         const double* d_ay) {
    Level& F = lv_[0];
    CUDA_CHECK(cudaMemcpy(F.ax, d_ax, (F.nx + 1) * F.ny * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(F.ay, d_ay, F.nx * (F.ny + 1) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    buildCoarseFaces();
}

void MultigridPreconditioner::smooth(Level& L, int sweeps,
                                     const double* d_alpha,
                                     const double* d_sigma,
                                     cudaStream_t stream) {
    const int px = (bcX_ == BCKind::Periodic) ? 1 : 0;
    const int py = (bcY_ == BCKind::Periodic) ? 1 : 0;
    const int nc = L.nx * L.ny;
    for (int s = 0; s < sweeps; ++s) {
        k_mg_residual<<<blocks(nc), 256, 0, stream>>>(
            L.t, L.r, L.z, L.ax, L.ay, L.nx, L.ny, L.inv_dx2, L.inv_dy2,
            d_alpha, d_sigma, px, py);
        k_mg_jacobi<<<blocks(nc), 256, 0, stream>>>(
            L.z, L.t, L.ax, L.ay, L.nx, L.ny, L.inv_dx2, L.inv_dy2,
            d_alpha, d_sigma, omega_);
        CUDA_CHECK(cudaGetLastError());
    }
}

void MultigridPreconditioner::vcycle(int l, const double* d_alpha,
                                     const double* d_sigma,
                                     cudaStream_t stream) {
    Level& L = lv_[l];
    const int px = (bcX_ == BCKind::Periodic) ? 1 : 0;
    const int py = (bcY_ == BCKind::Periodic) ? 1 : 0;
    const int nc = L.nx * L.ny;

    if (l == static_cast<int>(lv_.size()) - 1) {
        smooth(L, coarseSweeps_, d_alpha, d_sigma, stream);
        return;
    }

    smooth(L, nPre_, d_alpha, d_sigma, stream);

    k_mg_residual<<<blocks(nc), 256, 0, stream>>>(
        L.t, L.r, L.z, L.ax, L.ay, L.nx, L.ny, L.inv_dx2, L.inv_dy2,
        d_alpha, d_sigma, px, py);
    CUDA_CHECK(cudaGetLastError());

    Level& C = lv_[l + 1];
    k_mg_restrict<<<blocks(C.nx * C.ny), 256, 0, stream>>>(
        C.r, L.t, C.nx, C.ny, L.nx, L.ny, px, py);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(C.z, 0, C.nx * C.ny * sizeof(Real), stream));

    vcycle(l + 1, d_alpha, d_sigma, stream);

    k_mg_prolong_add<<<blocks(nc), 256, 0, stream>>>(
        L.z, C.z, L.nx, L.ny, C.nx, C.ny, px, py);
    CUDA_CHECK(cudaGetLastError());

    smooth(L, nPost_, d_alpha, d_sigma, stream);
}

void MultigridPreconditioner::apply(ScalarField& r, ScalarField& z,
                                    const double* d_alpha,
                                    const double* d_sigma,
                                    cudaStream_t stream) {
    if (!facesReady_)
        throw std::runtime_error(
            "MultigridPreconditioner::apply: call setup()/setupFaces() first");
    Level& F = lv_[0];
    if (r.mesh.n[0] != F.nx || r.mesh.n[1] != F.ny)
        throw std::invalid_argument(
            "MultigridPreconditioner::apply: field mesh differs from setup");

    const int nc = F.nx * F.ny;
    const int sx = r.storedDims[0];
    const int planeOff = sx * r.storedDims[1] * r.ghost;
    k_mg_pack<<<blocks(nc), 256, 0, stream>>>(F.r, r.d_curr, F.nx, F.ny,
                                              sx, r.ghost, planeOff);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(F.z, 0, nc * sizeof(Real), stream));

    vcycle(0, d_alpha, d_sigma, stream);

    k_mg_unpack<<<blocks(nc), 256, 0, stream>>>(z.d_curr, F.z, F.nx, F.ny,
                                                sx, z.ghost, planeOff);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
