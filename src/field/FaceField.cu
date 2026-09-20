#include "field/FaceField.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace PhiX {

namespace {

Centering centFromAxis(int ax) {
    if (ax == 0) return Centering::FACE_X;
    if (ax == 1) return Centering::FACE_Y;
    if (ax == 2) return Centering::FACE_Z;
    throw std::invalid_argument("FaceField: normalAxis must be 0, 1, or 2");
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
FaceField::FaceField(const Mesh& mesh_, int ax, const std::string& name_, int ghost_)
    : name(name_)
    , layout(mesh_, ghost_, centFromAxis(ax))
    , normalAxis(ax)
    , mesh(mesh_)
    , ghost(ghost_)
    , storedSize(layout.storedSize)
{
    if (ax < 0 || ax >= mesh_.dim)
        throw std::invalid_argument(
            "FaceField: normalAxis " + std::to_string(ax)
            + " out of range for dim=" + std::to_string(mesh_.dim));

    for (int d = 0; d < 3; ++d) storedDims[d] = layout.storedDims[d];
    data.assign(storedSize, 0.0);
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------
FaceField::FaceField(FaceField&& o) noexcept
    : name(std::move(o.name))
    , layout(o.layout)
    , normalAxis(o.normalAxis)
    , mesh(o.mesh)
    , ghost(o.ghost)
    , storedSize(o.storedSize)
    , data(std::move(o.data))
    , d_data(o.d_data)
    , ownsDeviceMemory_(o.ownsDeviceMemory_)
{
    for (int d = 0; d < 3; ++d) storedDims[d] = o.storedDims[d];
    o.d_data = nullptr;
}

FaceField& FaceField::operator=(FaceField&& o) noexcept {
    if (this == &o) return *this;
    if (ownsDeviceMemory_ && d_data) cudaFree(d_data);
    name            = std::move(o.name);
    layout          = o.layout;
    normalAxis      = o.normalAxis;
    // mesh is a const reference, cannot rebind — UB if meshes differ, but
    // move assign is only used when the mesh pointer stays the same.
    ghost           = o.ghost;
    storedSize      = o.storedSize;
    for (int d = 0; d < 3; ++d) storedDims[d] = o.storedDims[d];
    data            = std::move(o.data);
    d_data          = o.d_data;
    ownsDeviceMemory_ = o.ownsDeviceMemory_;
    o.d_data        = nullptr;
    return *this;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FaceField::~FaceField() {
    if (ownsDeviceMemory_ && d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
void FaceField::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------
void FaceField::allocDevice() {
    if (d_data) return;  // already allocated
    CUDA_CHECK(cudaMalloc(&d_data, storedSize * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_data, 0, storedSize * sizeof(Real)));
}

void FaceField::freeDevice() {
    if (ownsDeviceMemory_ && d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}

void FaceField::uploadToDevice() const {
    if (!d_data)
        throw std::runtime_error("FaceField::uploadToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_data, data.data(),
                          storedSize * sizeof(Real),
                          cudaMemcpyHostToDevice));
}

void FaceField::downloadFromDevice() {
    if (!d_data)
        throw std::runtime_error("FaceField::downloadFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(data.data(), d_data,
                          storedSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));
}

} // namespace PhiX
