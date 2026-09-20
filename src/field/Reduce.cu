#include "field/Reduce.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {
namespace reduce {

namespace {

// ---------------------------------------------------------------------------
// Gather functors — map a physical linear index p in [0, nx*ny*nz) to the
// stored (halo-padded) index and fetch/transform the value.  Used through
// thrust::transform_iterator so cub::DeviceReduce never touches ghost cells.
// ---------------------------------------------------------------------------
struct GatherBase {
    const Real* data;
    int nx, ny, sx, sy, g;

    __host__ __device__ double fetch(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        return data[c];
    }
};

struct GatherValue : GatherBase {
    __host__ __device__ double operator()(int p) const { return fetch(p); }
};
struct GatherAbs : GatherBase {
    __host__ __device__ double operator()(int p) const { return fabs(fetch(p)); }
};
struct GatherSq : GatherBase {
    __host__ __device__ double operator()(int p) const {
        const double v = fetch(p);
        return v * v;
    }
};
struct GatherNonFinite : GatherBase {
    __host__ __device__ int operator()(int p) const {
        return isfinite(fetch(p)) ? 0 : 1;
    }
};
struct GatherDot : GatherBase {
    const Real* data2;
    __host__ __device__ double operator()(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        return static_cast<double>(data[c]) * static_cast<double>(data2[c]);
    }
};

// ---------------------------------------------------------------------------
// Cached device scratch: CUB temp storage (grow-only) + one 8-byte result
// slot.  NOT freed in a static destructor (the CUDA context may already be
// gone at exit) — freeScratch() releases explicitly.
// ---------------------------------------------------------------------------
struct Scratch {
    void*       d_temp     = nullptr;
    std::size_t temp_bytes = 0;
    void*       d_out      = nullptr;   // 8 bytes: double or int result
};
Scratch g_scratch;

void ensureScratch(std::size_t bytes) {
    if (!g_scratch.d_out)
        CUDA_CHECK(cudaMalloc(&g_scratch.d_out, sizeof(double)));
    if (bytes > g_scratch.temp_bytes) {
        if (g_scratch.d_temp) CUDA_CHECK(cudaFree(g_scratch.d_temp));
        CUDA_CHECK(cudaMalloc(&g_scratch.d_temp, bytes));
        g_scratch.temp_bytes = bytes;
    }
}

template<typename Gather>
Gather makeGather(const ScalarField& f, const char* fn) {
    if (!f.d_curr)
        throw std::runtime_error(std::string(fn) + ": field '" + f.name
                                 + "' has no device allocation");
    Gather op{};
    op.data = f.d_curr;
    op.nx = f.mesh.n[0];
    op.ny = f.mesh.n[1];
    op.sx = f.storedDims[0];
    op.sy = f.storedDims[1];
    op.g  = f.ghost;
    return op;
}

// Run one CUB device reduction over the physical cells and return the result.
// Op is invoked as op(d_temp, bytes, iterator, d_out, n).
template<typename T, typename Gather, typename CubOp>
T runReduce(const ScalarField& f, const char* fn, CubOp cubOp) {
    const Gather gather = makeGather<Gather>(f, fn);
    const int n = f.mesh.n[0] * f.mesh.n[1] * f.mesh.n[2];

    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);

    std::size_t bytes = 0;
    CUDA_CHECK(cubOp(nullptr, bytes, it, static_cast<T*>(g_scratch.d_out), n));
    ensureScratch(bytes);
    CUDA_CHECK(cubOp(g_scratch.d_temp, bytes, it,
                     static_cast<T*>(g_scratch.d_out), n));

    T h{};
    CUDA_CHECK(cudaMemcpy(&h, g_scratch.d_out, sizeof(T),
                          cudaMemcpyDeviceToHost));
    return h;
}

// cub::DeviceReduce entry points wrapped as plain callables (the member
// templates cannot be passed directly as template-template arguments).
struct CubMax {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Max(t, b, it, out, n);
    }
};
struct CubMin {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Min(t, b, it, out, n);
    }
};
struct CubSum {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Sum(t, b, it, out, n);
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

double fieldMax(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldMax", CubMax{});
}

double fieldMin(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldMin", CubMin{});
}

double fieldMaxAbs(const ScalarField& f) {
    return runReduce<double, GatherAbs>(f, "reduce::fieldMaxAbs", CubMax{});
}

double fieldSum(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldSum", CubSum{});
}

double fieldSumSq(const ScalarField& f) {
    return runReduce<double, GatherSq>(f, "reduce::fieldSumSq", CubSum{});
}

double fieldL2(const ScalarField& f) {
    return std::sqrt(runReduce<double, GatherSq>(f, "reduce::fieldL2", CubSum{}));
}

bool fieldHasNonFinite(const ScalarField& f) {
    return runReduce<int, GatherNonFinite>(f, "reduce::fieldHasNonFinite",
                                           CubMax{}) != 0;
}

// Enqueue the CUB dot-product reduction with the result left in d_out.
static void enqueueDot(const ScalarField& a, const ScalarField& b,
                       double* d_out, const char* fn,
                       cudaStream_t stream = nullptr) {
    if (a.storedSize != b.storedSize || a.ghost != b.ghost)
        throw std::invalid_argument(
            std::string(fn) + ": fields '" + a.name + "' and '" + b.name
            + "' have different layouts");
    if (!b.d_curr)
        throw std::runtime_error(std::string(fn) + ": field '" + b.name
                                 + "' has no device allocation");
    GatherDot gather = makeGather<GatherDot>(a, fn);
    gather.data2 = b.d_curr;

    const int n = a.mesh.n[0] * a.mesh.n[1] * a.mesh.n[2];
    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);

    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, it, d_out, n, stream));
    ensureScratch(bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(g_scratch.d_temp, bytes, it, d_out, n,
                                      stream));
}

double fieldDot(const ScalarField& a, const ScalarField& b) {
    ensureScratch(0);   // d_out must exist BEFORE its value is passed on
    enqueueDot(a, b, static_cast<double*>(g_scratch.d_out),
               "reduce::fieldDot");
    double h = 0.0;
    CUDA_CHECK(cudaMemcpy(&h, g_scratch.d_out, sizeof(double),
                          cudaMemcpyDeviceToHost));
    return h;
}

void fieldDotAsync(const ScalarField& a, const ScalarField& b,
                   double* d_out, cudaStream_t stream) {
    if (!g_scratch.d_out)         // ensureScratch touches d_out lazily
        CUDA_CHECK(cudaMalloc(&g_scratch.d_out, sizeof(double)));
    enqueueDot(a, b, d_out, "reduce::fieldDotAsync", stream);
}

// ---------------------------------------------------------------------------
// |∇f|² sum (CD2) — fused gather, no scratch field
// ---------------------------------------------------------------------------
namespace {
struct GatherGradSq : GatherBase {
    int  sz;      // z stride (sx*sy)
    int  dim;
    Real i2dx, i2dy, i2dz;   // 0.5/d
    __host__ __device__ double operator()(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        double gx = static_cast<double>((data[c + 1] - data[c - 1]) * i2dx);
        double s2 = gx * gx;
        if (dim >= 2) {
            const double gy = static_cast<double>(
                (data[c + sx] - data[c - sx]) * i2dy);
            s2 += gy * gy;
        }
        if (dim >= 3) {
            const double gz = static_cast<double>(
                (data[c + sz] - data[c - sz]) * i2dz);
            s2 += gz * gz;
        }
        return s2;
    }
};
} // namespace

double fieldGradSq(const ScalarField& f) {
    if (f.ghost < 1)
        throw std::invalid_argument("reduce::fieldGradSq: ghost >= 1 required");
    GatherGradSq gather = makeGather<GatherGradSq>(f, "reduce::fieldGradSq");
    gather.sz   = f.storedDims[0] * f.storedDims[1];
    gather.dim  = f.mesh.dim;
    gather.i2dx = static_cast<Real>(0.5 / f.mesh.d[0]);
    gather.i2dy = (f.mesh.dim >= 2)
        ? static_cast<Real>(0.5 / f.mesh.d[1]) : Real(0);
    gather.i2dz = (f.mesh.dim >= 3)
        ? static_cast<Real>(0.5 / f.mesh.d[2]) : Real(0);

    const int n = f.mesh.n[0] * f.mesh.n[1] * f.mesh.n[2];
    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);
    std::size_t bytes = 0;
    ensureScratch(0);
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, it,
                                      static_cast<double*>(g_scratch.d_out), n));
    ensureScratch(bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(g_scratch.d_temp, bytes, it,
                                      static_cast<double*>(g_scratch.d_out), n));
    double h = 0.0;
    CUDA_CHECK(cudaMemcpy(&h, g_scratch.d_out, sizeof(double),
                          cudaMemcpyDeviceToHost));
    return h;
}

namespace detail {
void* scratchTemp(std::size_t bytes) {
    ensureScratch(bytes);
    return g_scratch.d_temp;
}
double* scratchOut() {
    ensureScratch(0);
    return static_cast<double*>(g_scratch.d_out);
}
} // namespace detail

void freeScratch() {
    if (g_scratch.d_temp) CUDA_CHECK(cudaFree(g_scratch.d_temp));
    if (g_scratch.d_out)  CUDA_CHECK(cudaFree(g_scratch.d_out));
    g_scratch = Scratch{};
}

} // namespace reduce
} // namespace PhiX
