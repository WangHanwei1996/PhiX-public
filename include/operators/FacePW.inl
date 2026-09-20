// ---------------------------------------------------------------------------
// FacePW.inl — Template definitions for facePW / facePWGPU.
// Included automatically by FacePW.h.  Do NOT include directly.
// Requires nvcc (contains __global__ kernel templates).
// ---------------------------------------------------------------------------

#pragma once

#include "core/CudaCheck.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace PhiX {

// ===========================================================================
// Helpers
// ===========================================================================

// Face linear index — mirrors face_idx() in FaceOps.cu.
// Normal axis has no ghost offset; tangential axes have +ghost offset.
__host__ __device__ inline
int facepw_idx(int i, int j, int k, int ax, int sx, int sy, int g)
{
    int si = (ax == 0) ? i : (i + g);
    int sj = (ax == 1) ? j : (j + g);
    int sk = (ax == 2) ? k : (k + g);
    return si + sx * (sj + sy * sk);
}

// ===========================================================================
// CUDA kernels
// ===========================================================================

// 1-field
template<typename Fn>
__global__ void kernel_facepw1(Real* out, const Real* a,
                                Fn fn,
                                int lim0, int lim1, int lim2,
                                int ax, int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lim0 * lim1 * lim2) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int idx = facepw_idx(i, j, k, ax, sx, sy, g);
    out[idx] = fn(a[idx]);
}

// 2-field
template<typename Fn>
__global__ void kernel_facepw2(Real* out,
                                const Real* a, const Real* b,
                                Fn fn,
                                int lim0, int lim1, int lim2,
                                int ax, int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lim0 * lim1 * lim2) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int idx = facepw_idx(i, j, k, ax, sx, sy, g);
    out[idx] = fn(a[idx], b[idx]);
}

// 3-field
template<typename Fn>
__global__ void kernel_facepw3(Real* out,
                                const Real* a, const Real* b, const Real* c,
                                Fn fn,
                                int lim0, int lim1, int lim2,
                                int ax, int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lim0 * lim1 * lim2) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int idx = facepw_idx(i, j, k, ax, sx, sy, g);
    out[idx] = fn(a[idx], b[idx], c[idx]);
}

// 4-field
template<typename Fn>
__global__ void kernel_facepw4(Real* out,
                                const Real* a, const Real* b,
                                const Real* c, const Real* d,
                                Fn fn,
                                int lim0, int lim1, int lim2,
                                int ax, int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lim0 * lim1 * lim2) return;

    int i = tid % lim0;
    int j = (tid / lim0) % lim1;
    int k = tid / (lim0 * lim1);

    int idx = facepw_idx(i, j, k, ax, sx, sy, g);
    out[idx] = fn(a[idx], b[idx], c[idx], d[idx]);
}

// ===========================================================================
// Validation helpers (shared by CPU and GPU paths)
// ===========================================================================

namespace detail {

inline void checkFaceCompat(const FaceField& out,
                             const FaceField& a,
                             const char* where)
{
    if (out.normalAxis != a.normalAxis)
        throw std::invalid_argument(
            std::string(where) + ": normalAxis mismatch");
    if (out.storedDims[0] != a.storedDims[0] ||
        out.storedDims[1] != a.storedDims[1] ||
        out.storedDims[2] != a.storedDims[2])
        throw std::invalid_argument(
            std::string(where) + ": storedDims mismatch");
}

inline void checkFaceCompat3(const FaceField& out,
                              const FaceField& a,
                              const FaceField& b,
                              const FaceField& c,
                              const char* where)
{
    checkFaceCompat(out, a, where);
    checkFaceCompat(out, b, where);
    checkFaceCompat(out, c, where);
}

} // namespace detail

// ===========================================================================
// CPU implementations
// ===========================================================================

template<typename Fn>
void facePW(FaceField& out, const FaceField& a, Fn fn)
{
    detail::checkFaceCompat(out, a, "facePW(1)");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    // Loop bounds: normal axis → n+1 faces; tangential axes → n physical cells
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];

    const Real* ad = a.data.data();
    Real*       od = out.data.data();

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int idx = facepw_idx(i, j, k, ax, sx, sy, g);
        od[idx] = fn(ad[idx]);
    }
}

template<typename Fn>
void facePW(FaceField& out, const FaceField& a, const FaceField& b, Fn fn)
{
    detail::checkFaceCompat(out, a, "facePW(2)");
    detail::checkFaceCompat(out, b, "facePW(2)");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];

    const Real* ad = a.data.data();
    const Real* bd = b.data.data();
    Real*       od = out.data.data();

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int idx = facepw_idx(i, j, k, ax, sx, sy, g);
        od[idx] = fn(ad[idx], bd[idx]);
    }
}

template<typename Fn>
void facePW(FaceField& out,
            const FaceField& a, const FaceField& b, const FaceField& c,
            Fn fn)
{
    detail::checkFaceCompat3(out, a, b, c, "facePW(3)");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];

    const Real* ad = a.data.data();
    const Real* bd = b.data.data();
    const Real* cd = c.data.data();
    Real*       od = out.data.data();

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int idx = facepw_idx(i, j, k, ax, sx, sy, g);
        od[idx] = fn(ad[idx], bd[idx], cd[idx]);
    }
}

template<typename Fn>
void facePW(FaceField& out,
            const FaceField& a, const FaceField& b, const FaceField& c,
            const FaceField& d, Fn fn)
{
    detail::checkFaceCompat3(out, a, b, c, "facePW(4)");
    detail::checkFaceCompat(out, d, "facePW(4)");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];

    const Real* ad = a.data.data();
    const Real* bd = b.data.data();
    const Real* cd = c.data.data();
    const Real* dd = d.data.data();
    Real*       od = out.data.data();

    for (int k = 0; k < lim[2]; ++k)
    for (int j = 0; j < lim[1]; ++j)
    for (int i = 0; i < lim[0]; ++i) {
        int idx = facepw_idx(i, j, k, ax, sx, sy, g);
        od[idx] = fn(ad[idx], bd[idx], cd[idx], dd[idx]);
    }
}

// ===========================================================================
// GPU implementations
// ===========================================================================

template<typename Fn>
void facePWGPU(FaceField& out, const FaceField& a, Fn fn)
{
    detail::checkFaceCompat(out, a, "facePWGPU(1)");
    if (!a.d_data)   throw std::runtime_error("facePWGPU(1): a not on device");
    if (!out.d_data) throw std::runtime_error("facePWGPU(1): out not on device");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];
    const int total = lim[0] * lim[1] * lim[2];

    kernel_facepw1<Fn><<<(total + 255) / 256, 256>>>(
        out.d_data, a.d_data, fn,
        lim[0], lim[1], lim[2], ax, sx, sy, g);
    PHIX_KERNEL_CHECK("facePWGPU(1)");
}

template<typename Fn>
void facePWGPU(FaceField& out, const FaceField& a, const FaceField& b, Fn fn)
{
    detail::checkFaceCompat(out, a, "facePWGPU(2)");
    detail::checkFaceCompat(out, b, "facePWGPU(2)");
    if (!a.d_data)   throw std::runtime_error("facePWGPU(2): a not on device");
    if (!b.d_data)   throw std::runtime_error("facePWGPU(2): b not on device");
    if (!out.d_data) throw std::runtime_error("facePWGPU(2): out not on device");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];
    const int total = lim[0] * lim[1] * lim[2];

    kernel_facepw2<Fn><<<(total + 255) / 256, 256>>>(
        out.d_data, a.d_data, b.d_data, fn,
        lim[0], lim[1], lim[2], ax, sx, sy, g);
    PHIX_KERNEL_CHECK("facePWGPU(2)");
}

template<typename Fn>
void facePWGPU(FaceField& out,
               const FaceField& a, const FaceField& b, const FaceField& c,
               Fn fn)
{
    detail::checkFaceCompat3(out, a, b, c, "facePWGPU(3)");
    if (!a.d_data)   throw std::runtime_error("facePWGPU(3): a not on device");
    if (!b.d_data)   throw std::runtime_error("facePWGPU(3): b not on device");
    if (!c.d_data)   throw std::runtime_error("facePWGPU(3): c not on device");
    if (!out.d_data) throw std::runtime_error("facePWGPU(3): out not on device");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];
    const int total = lim[0] * lim[1] * lim[2];

    kernel_facepw3<Fn><<<(total + 255) / 256, 256>>>(
        out.d_data, a.d_data, b.d_data, c.d_data, fn,
        lim[0], lim[1], lim[2], ax, sx, sy, g);
    PHIX_KERNEL_CHECK("facePWGPU(3)");
}

template<typename Fn>
void facePWGPU(FaceField& out,
               const FaceField& a, const FaceField& b, const FaceField& c,
               const FaceField& d, Fn fn)
{
    detail::checkFaceCompat3(out, a, b, c, "facePWGPU(4)");
    detail::checkFaceCompat(out, d, "facePWGPU(4)");
    if (!a.d_data)   throw std::runtime_error("facePWGPU(4): a not on device");
    if (!b.d_data)   throw std::runtime_error("facePWGPU(4): b not on device");
    if (!c.d_data)   throw std::runtime_error("facePWGPU(4): c not on device");
    if (!d.d_data)   throw std::runtime_error("facePWGPU(4): d not on device");
    if (!out.d_data) throw std::runtime_error("facePWGPU(4): out not on device");

    const int ax = out.normalAxis;
    const int g  = out.ghost;
    int lim[3] = { out.mesh.n[0], out.mesh.n[1], out.mesh.n[2] };
    lim[ax] += 1;
    const int sx = out.storedDims[0], sy = out.storedDims[1];
    const int total = lim[0] * lim[1] * lim[2];

    kernel_facepw4<Fn><<<(total + 255) / 256, 256>>>(
        out.d_data, a.d_data, b.d_data, c.d_data, d.d_data, fn,
        lim[0], lim[1], lim[2], ax, sx, sy, g);
    PHIX_KERNEL_CHECK("facePWGPU(4)");
}

} // namespace PhiX
