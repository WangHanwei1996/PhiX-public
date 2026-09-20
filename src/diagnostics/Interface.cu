#include "diagnostics/Interface.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace PhiX {

namespace {

__global__ void kernel_gather_line(Real* out, const Real* f,
                                   int n, int base, int stride)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f[base + i * stride];
}

} // namespace

double interfacePosition(const ScalarField& f, int axis, int t0, int t1,
                         double level, bool scanFromHigh)
{
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("interfacePosition: axis out of range");
    if (!f.d_curr)
        throw std::runtime_error("interfacePosition: field not on device");

    const int n = f.mesh.n[axis];
    const int g = f.ghost;
    const int sx = f.storedDims[0], sy = f.storedDims[1];
    const int strides[3] = {1, sx, sx * sy};

    // stored offset of the line start: physical cell 0 along `axis`,
    // transverse indices (t0, t1) in ascending remaining-axis order
    int idx3[3] = {0, 0, 0};
    int t = 0;
    for (int a = 0; a < 3; ++a) {
        if (a == axis) continue;
        idx3[a] = (t++ == 0) ? t0 : t1;
    }
    const int lineBase = (idx3[0] + g)
                       + sx * ((idx3[1] + g) + sy * (idx3[2] + g));

    // gather the line into a small device buffer, copy to host
    static Real*       d_line = nullptr;
    static std::size_t cap = 0;
    if (static_cast<std::size_t>(n) > cap) {
        if (d_line) CUDA_CHECK(cudaFree(d_line));
        CUDA_CHECK(cudaMalloc(&d_line, n * sizeof(Real)));
        cap = static_cast<std::size_t>(n);
    }
    kernel_gather_line<<<(n + 255) / 256, 256>>>(
        d_line, f.d_curr, n, lineBase, strides[axis]);
    CUDA_CHECK(cudaGetLastError());
    std::vector<Real> line(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(line.data(), d_line, n * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    const int i0   = scanFromHigh ? n - 2 : 0;
    const int iEnd = scanFromHigh ? -1 : n - 1;
    const int step = scanFromHigh ? -1 : 1;
    for (int i = i0; i != iEnd; i += step) {
        const double a = static_cast<double>(line[static_cast<std::size_t>(i)]);
        const double b = static_cast<double>(line[static_cast<std::size_t>(i + 1)]);
        if ((a - level) * (b - level) <= 0.0 && a != b) {
            const double frac = (level - a) / (b - a);
            return f.mesh.coord(axis, i) + frac * f.mesh.d[axis];
        }
    }
    throw std::runtime_error("interfacePosition: level "
                             + std::to_string(level)
                             + " not crossed along the line");
}

} // namespace PhiX
