#include "field/ScalarField.h"
#include "core/CudaCheck.h"
#include "IO/FieldIO.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ScalarField::ScalarField(const Mesh& mesh_, const std::string& name_, int ghost_)
    : ScalarField(FieldLayout(mesh_, ghost_), name_)
{}

ScalarField::ScalarField(const FieldLayout& layout_, const std::string& name_)
    : name(name_)
    , layout(layout_)
    , mesh(layout.meshRef())
    , ghost(layout.ghost)
    , storedSize(layout.storedSize)
{
    storedDims[0] = layout.storedDims[0];
    storedDims[1] = layout.storedDims[1];
    storedDims[2] = layout.storedDims[2];

    curr.assign(storedSize, 0.0);
    prev.assign(storedSize, 0.0);
}

// ---------------------------------------------------------------------------
// Shell factory — wraps an externally owned device buffer.  CPU storage is
// left empty; ownsDeviceMemory_ = false so destructor does not free.
// ---------------------------------------------------------------------------
ScalarField ScalarField::makeShell(const Mesh& mesh_, int ghost_,
                                   Real* d_buf,
                                   const std::string& name_)
{
    ScalarField f(FieldLayout(mesh_, ghost_), name_);
    // Drop CPU buffers — a shell only exposes device data.
    f.curr.clear(); f.curr.shrink_to_fit();
    f.prev.clear(); f.prev.shrink_to_fit();
    f.d_curr            = d_buf;
    f.d_prev            = nullptr;
    f.ownsDeviceMemory_ = false;
    return f;
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

ScalarField::ScalarField(ScalarField&& other) noexcept
    : name(std::move(other.name))
    , layout(other.layout)
    , mesh(other.mesh)
    , ghost(other.ghost)
    , storedSize(other.storedSize)
    , curr(std::move(other.curr))
    , prev(std::move(other.prev))
    , d_curr(other.d_curr)
    , d_prev(other.d_prev)
    , ownsDeviceMemory_(other.ownsDeviceMemory_)
{
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    other.d_curr = nullptr;
    other.d_prev = nullptr;
    other.ownsDeviceMemory_ = false;
}

ScalarField& ScalarField::operator=(ScalarField&& other) noexcept {
    if (this == &other) return *this;
    freeDevice();

    name      = std::move(other.name);
    layout    = other.layout;
    // mesh is a const ref — cannot rebind, caller is responsible for lifetime
    ghost     = other.ghost;
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    storedSize = other.storedSize;
    curr      = std::move(other.curr);
    prev      = std::move(other.prev);
    d_curr    = other.d_curr;  other.d_curr = nullptr;
    d_prev    = other.d_prev;  other.d_prev = nullptr;
    ownsDeviceMemory_ = other.ownsDeviceMemory_;
    other.ownsDeviceMemory_ = false;
    return *this;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

ScalarField::~ScalarField() {
    freeDevice();
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void ScalarField::fill(double value) {
    fillCurr(value);
    fillPrev(value);
}

void ScalarField::fillCurr(double value) {
    std::fill(curr.begin(), curr.end(), value);
}

void ScalarField::fillPrev(double value) {
    std::fill(prev.begin(), prev.end(), value);
}

// ---------------------------------------------------------------------------
// Time-stepping
// ---------------------------------------------------------------------------

void ScalarField::advanceTimeLevelCPU() {
    if (!trackPrev) return;   // prev untracked → rotation is a no-op
    std::copy(curr.begin(), curr.end(), prev.begin());
}

void ScalarField::advanceTimeLevelGPU() {
    hostCurrStale = true;     // the device copy has advanced past the host
    if (!trackPrev) return;   // prev untracked → rotation is a no-op
    if (!deviceAllocated() || !d_prev)
        throw std::runtime_error("ScalarField::advanceTimeLevelGPU: device "
                                 "not allocated (or trackPrev set after "
                                 "allocDevice)");
    CUDA_CHECK(cudaMemcpy(d_prev, d_curr,
                          storedSize * sizeof(Real),
                          cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------

void ScalarField::allocDevice() {
    if (deviceAllocated()) return;
    const std::size_t bytes = storedSize * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_curr, bytes));
    CUDA_CHECK(cudaMemset(d_curr, 0, bytes));
    // The prev buffer costs a full field of device memory and a full D2D
    // copy per step — only pay for it when the app opted in.
    if (trackPrev) {
        CUDA_CHECK(cudaMalloc(&d_prev, bytes));
        CUDA_CHECK(cudaMemset(d_prev, 0, bytes));
    }
}

void ScalarField::freeDevice() {
    if (ownsDeviceMemory_) {
        if (d_curr) { cudaFree(d_curr); }
        if (d_prev) { cudaFree(d_prev); }
    }
    d_curr = nullptr;
    d_prev = nullptr;
}

void ScalarField::uploadCurrToDevice() const {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::uploadCurrToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_curr, curr.data(),
                          storedSize * sizeof(Real),
                          cudaMemcpyHostToDevice));
    hostCurrStale = false;
}

void ScalarField::uploadPrevToDevice() const {
    if (!d_prev)
        throw std::runtime_error("ScalarField::uploadPrevToDevice: prev not "
                                 "tracked on device (set trackPrev = true "
                                 "before allocDevice)");
    CUDA_CHECK(cudaMemcpy(d_prev, prev.data(),
                          storedSize * sizeof(Real),
                          cudaMemcpyHostToDevice));
}

void ScalarField::uploadAllToDevice() const {
    uploadCurrToDevice();
    if (trackPrev) uploadPrevToDevice();
}

void ScalarField::downloadCurrFromDevice() {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::downloadCurrFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(curr.data(), d_curr,
                          storedSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    hostCurrStale = false;
}

void ScalarField::downloadPrevFromDevice() {
    if (!d_prev)
        throw std::runtime_error("ScalarField::downloadPrevFromDevice: prev "
                                 "not tracked on device (set trackPrev = "
                                 "true before allocDevice)");
    CUDA_CHECK(cudaMemcpy(prev.data(), d_prev,
                          storedSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));
}

void ScalarField::downloadAllFromDevice() {
    downloadCurrFromDevice();
    if (trackPrev) downloadPrevFromDevice();
}

// ---------------------------------------------------------------------------
// IO  (delegated to IO module)
// ---------------------------------------------------------------------------

void ScalarField::write(const std::string& path, FieldFormat fmt,
                        double coordScale) const {
    if (deviceAllocated() && hostCurrStale)
        warnOnce("stale-write:" + name,
                 "ScalarField::write: writing the HOST copy of field '" + name
                 + "' but the device copy has advanced — call "
                 "downloadCurrFromDevice() before write()");
    IO::writeField(*this, path, fmt, coordScale);
}

ScalarField ScalarField::readFromFile(const Mesh& mesh,
                                      const std::string& path,
                                      int ghost) {
    return IO::readScalarField(mesh, path, ghost);
}

// ---------------------------------------------------------------------------
// IO helper for print(): physical (i,j,k) -> stored flat index
// ---------------------------------------------------------------------------
static inline int physIdx(const ScalarField& f, int i, int j, int k) {
    return f.layout.index(i, j, k);
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void ScalarField::print() const {
    std::cout << "=== ScalarField ===\n";
    std::cout << "  name       : " << name << "\n";
    std::cout << "  ghost      : " << ghost << "\n";
    std::cout << "  storedDims : "
              << storedDims[0] << " x "
              << storedDims[1] << " x "
              << storedDims[2] << "  (" << storedSize << " cells)\n";
    std::cout << "  device     : " << (deviceAllocated() ? "allocated" : "not allocated") << "\n";

    // Min / max / mean of curr (physical cells only)
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();
    double vsum = 0.0;
    std::size_t count = 0;

    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i) {
                double v = curr[physIdx(*this, i, j, k)];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
                vsum += v;
                ++count;
            }

    if (count > 0) {
        std::cout << "  curr  min  : " << vmin << "\n";
        std::cout << "  curr  max  : " << vmax << "\n";
        std::cout << "  curr  mean : " << vsum / static_cast<double>(count) << "\n";
    }
}

} // namespace PhiX
