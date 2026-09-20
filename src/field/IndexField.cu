#include "field/IndexField.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// IndexField
// ---------------------------------------------------------------------------

IndexField::IndexField(const Mesh& mesh_, const std::string& name_, int ghost_)
    : name(name_)
    , layout(mesh_, ghost_)
    , mesh(mesh_)
    , ghost(ghost_)
    , storedSize(layout.storedSize)
{
    storedDims[0] = layout.storedDims[0];
    storedDims[1] = layout.storedDims[1];
    storedDims[2] = layout.storedDims[2];
    curr.assign(storedSize, -1);
}

IndexField::IndexField(IndexField&& other) noexcept
    : name(std::move(other.name))
    , layout(other.layout)
    , mesh(other.mesh)
    , ghost(other.ghost)
    , storedSize(other.storedSize)
    , curr(std::move(other.curr))
    , d_curr(other.d_curr)
{
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    other.d_curr = nullptr;
}

IndexField& IndexField::operator=(IndexField&& other) noexcept {
    if (this == &other) return *this;
    freeDevice();

    name   = std::move(other.name);
    layout = other.layout;
    // mesh is a const ref — cannot rebind, caller is responsible for lifetime
    ghost  = other.ghost;
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    storedSize = other.storedSize;
    curr   = std::move(other.curr);
    d_curr = other.d_curr;  other.d_curr = nullptr;
    return *this;
}

IndexField::~IndexField() {
    freeDevice();
}

void IndexField::fill(int32_t value) {
    curr.assign(storedSize, value);
}

void IndexField::allocDevice() {
    if (d_curr) return;
    CUDA_CHECK(cudaMalloc(&d_curr, storedBytes()));
}

void IndexField::freeDevice() {
    if (d_curr) { cudaFree(d_curr); d_curr = nullptr; }
}

void IndexField::uploadToDevice() const {
    if (!d_curr)
        throw std::runtime_error(
            "IndexField::uploadToDevice: call allocDevice() first");
    CUDA_CHECK(cudaMemcpy(d_curr, curr.data(), storedBytes(),
                          cudaMemcpyHostToDevice));
}

void IndexField::downloadFromDevice() {
    if (!d_curr)
        throw std::runtime_error(
            "IndexField::downloadFromDevice: device buffer not allocated");
    CUDA_CHECK(cudaMemcpy(curr.data(), d_curr, storedBytes(),
                          cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// DeviceTable
// ---------------------------------------------------------------------------

DeviceTable::DeviceTable(const std::vector<double>& host)
    : n_(host.size())
{
    if (n_ == 0)
        throw std::invalid_argument("DeviceTable: empty host table");
    CUDA_CHECK(cudaMalloc(&d_, n_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_, host.data(), n_ * sizeof(double),
                          cudaMemcpyHostToDevice));
}

DeviceTable::DeviceTable(DeviceTable&& other) noexcept
    : d_(other.d_), n_(other.n_)
{
    other.d_ = nullptr;
    other.n_ = 0;
}

DeviceTable& DeviceTable::operator=(DeviceTable&& other) noexcept {
    if (this == &other) return *this;
    if (d_) cudaFree(d_);
    d_ = other.d_;  other.d_ = nullptr;
    n_ = other.n_;  other.n_ = 0;
    return *this;
}

DeviceTable::~DeviceTable() {
    if (d_) cudaFree(d_);
}

// ---------------------------------------------------------------------------
// gatherGPU
// ---------------------------------------------------------------------------

namespace {

__global__ void kernel_gather(
        Real*          out,
        const int32_t* idx,
        const double*  table,
        double         fallback,
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
    out[c] = (id >= 0) ? table[id] : fallback;
}

} // anonymous namespace

void gatherGPU(ScalarField& out, const IndexField& idx,
               const double* d_table, double fallback,
               cudaStream_t stream)
{
    detail::checkIndexPrimitive(idx, out, "gatherGPU");
    if (!d_table)
        throw std::invalid_argument("gatherGPU: d_table is null");

    const int nx = idx.mesh.n[0], ny = idx.mesh.n[1], nz = idx.mesh.n[2];
    const int total   = nx * ny * nz;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    kernel_gather<<<blocks, threads, 0, stream>>>(
        out.d_curr, idx.d_curr, d_table, fallback,
        nx, ny, nz, idx.storedDims[0], idx.storedDims[1], idx.ghost);
    detail::checkKernelLaunch("gatherGPU");
}

} // namespace PhiX
