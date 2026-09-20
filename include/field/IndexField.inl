// ---------------------------------------------------------------------------
// IndexField.inl — Template primitives for IndexField (adopt / gatherPW).
// Included automatically by IndexField.h.  Do NOT include directly.
// Requires nvcc (contains __global__ kernel templates, like TermPW.inl).
// ---------------------------------------------------------------------------

#pragma once

#include "core/CudaCheck.h"
#include "field/ScalarField.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>

#ifndef PHIX_FN
#define PHIX_FN [=] __host__ __device__
#endif

namespace PhiX {

namespace detail {

// Shared launch-time validation for the IndexField primitives.
inline void checkIndexPrimitive(const IndexField& idx, const ScalarField& f,
                                const char* who) {
    if (idx.mesh.n[0] != f.mesh.n[0] ||
        idx.mesh.n[1] != f.mesh.n[1] ||
        idx.mesh.n[2] != f.mesh.n[2] || idx.ghost != f.ghost)
        throw std::invalid_argument(std::string(who) +
            ": IndexField and ScalarField must share mesh dims and ghost");
    if (!idx.deviceAllocated() || !f.d_curr)
        throw std::runtime_error(std::string(who) +
            ": a field is not on device (allocDevice + upload first)");
}

// checkKernelLaunch(who) now comes from core/CudaCheck.h (same
// PhiX::detail namespace) and throws DeviceError instead of runtime_error.

} // namespace detail

// ---------------------------------------------------------------------------
// kernel: neighbourhood-argmax conditional adoption (in-place on idx).
// One thread per physical cell; the full 8- (2D) / 26- (3D) neighbourhood is
// scanned in dk→dj→di order with a strict '>' comparison, so ties resolve to
// the first neighbour in scan order (same convention as PoolSectionGPU
// k_gid).  Ghost cells never participate: neighbours are clipped to the
// physical domain.
// ---------------------------------------------------------------------------
template<typename Pred>
__global__ void kernel_adopt_from_neighbor(
        int32_t*       idx,
        const Real*    score,
        Pred           arrived,
        Real*          out,        // may be nullptr (no table gather)
        const double*  table,      // must be valid when out != nullptr
        int32_t        minSourceId,// only neighbours with id >= this propagate
        int nx, int ny, int nz,
        int sx, int sy, int g, int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    if (idx[c] >= 0) return;          // already assigned
    if (!arrived(score[c])) return;   // front not here yet

    double  best   = -DBL_MAX;
    int32_t bestId = -1;
    const int dkR = (dim >= 3) ? 1 : 0;
    const int djR = (dim >= 2) ? 1 : 0;
    for (int dk = -dkR; dk <= dkR; ++dk)
    for (int dj = -djR; dj <= djR; ++dj)
    for (int di = -1;   di <= 1;   ++di) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        const int ii = i + di, jj = j + dj, kk = k + dk;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz)
            continue;
        const int n = (ii + g) + sx * ((jj + g) + sy * (kk + g));
        if (idx[n] >= minSourceId && score[n] > best) { best = score[n]; bestId = idx[n]; }
    }
    if (bestId >= 0) {
        idx[c] = bestId;
        if (out) out[c] = table[bestId];
    }
}

// ---------------------------------------------------------------------------
// adoptFromNeighborGPU — front-adoption primitive.
//
// For every UNASSIGNED physical cell (idx[c] < 0) where `arrived(score[c])`
// is true, adopt the index of the maximum-score assigned neighbour (if any).
// Runs in-place on idx.d_curr: assignments made within the same launch may
// or may not be visible to other cells — the standard benign race of
// front-adoption schemes where the front advances ≪ 1 cell per step.
//
//   adoptFromNeighborGPU(gid, phi,
//       PHIX_FN (double p) { return p >= -0.995; });
//
// The (out, d_table) overload additionally writes out[c] = d_table[newId]
// at the moment of adoption (per-grain orientation field etc.), fusing the
// table gather into the same kernel — the PoolSectionGPU k_gid pattern.
//
// minSourceId (default 0 = any assigned neighbour) restricts which grains
// may PROPAGATE: only neighbours with id >= minSourceId are adoption
// sources.  This is the Pool-family "substrate grains (id < firstSeed) are
// frozen decoration, only seed grains grow" rule.
// ---------------------------------------------------------------------------
template<typename Pred>
void adoptFromNeighborGPU(IndexField& idx, const ScalarField& score,
                          Pred arrived,
                          ScalarField* out = nullptr,
                          const double* d_table = nullptr,
                          int32_t minSourceId = 0,
                          cudaStream_t stream = nullptr)
{
    detail::checkIndexPrimitive(idx, score, "adoptFromNeighborGPU");
    if (out) {
        detail::checkIndexPrimitive(idx, *out, "adoptFromNeighborGPU(out)");
        if (!d_table)
            throw std::invalid_argument(
                "adoptFromNeighborGPU: out field given but d_table is null");
    }

    const int nx = idx.mesh.n[0], ny = idx.mesh.n[1], nz = idx.mesh.n[2];
    const int total   = nx * ny * nz;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    kernel_adopt_from_neighbor<Pred><<<blocks, threads, 0, stream>>>(
        idx.d_curr, score.d_curr, arrived,
        out ? out->d_curr : nullptr, d_table, minSourceId,
        nx, ny, nz, idx.storedDims[0], idx.storedDims[1], idx.ghost,
        idx.mesh.dim);
    detail::checkKernelLaunch("adoptFromNeighborGPU");
}

// ---------------------------------------------------------------------------
// kernel + gatherPWGPU — table gather with user functor and one aux field:
//   out[c] = fn(idx[c], idx[c] >= 0 ? table[idx[c]] : 0.0, aux[c])
// fn runs on EVERY physical cell (also unassigned ones — check the id
// argument inside the functor).  Use PHIX_FN for fn.
// ---------------------------------------------------------------------------
template<typename Fn>
__global__ void kernel_gather_pw(
        Real*          out,
        const int32_t* idx,
        const double*  table,
        const Real*    aux,
        Fn             fn,
        int nx, int ny, int nz,
        int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    const int32_t id = idx[c];
    out[c] = fn(id, (id >= 0) ? table[id] : 0.0, aux[c]);
}

template<typename Fn>
void gatherPWGPU(ScalarField& out, const IndexField& idx,
                 const double* d_table, const ScalarField& aux, Fn fn,
                 cudaStream_t stream = nullptr)
{
    detail::checkIndexPrimitive(idx, out, "gatherPWGPU(out)");
    detail::checkIndexPrimitive(idx, aux, "gatherPWGPU(aux)");
    if (!d_table)
        throw std::invalid_argument("gatherPWGPU: d_table is null");

    const int nx = idx.mesh.n[0], ny = idx.mesh.n[1], nz = idx.mesh.n[2];
    const int total   = nx * ny * nz;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    kernel_gather_pw<Fn><<<blocks, threads, 0, stream>>>(
        out.d_curr, idx.d_curr, d_table, aux.d_curr, fn,
        nx, ny, nz, idx.storedDims[0], idx.storedDims[1], idx.ghost);
    detail::checkKernelLaunch("gatherPWGPU");
}

} // namespace PhiX
