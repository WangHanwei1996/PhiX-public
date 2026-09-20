#pragma once

#include "field/FieldLayout.h"
#include "mesh/Mesh.h"

#include <string>
#include "core/Real.h"
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// FaceField
//
// Stores a scalar quantity defined at cell faces normal to `axis`.
// The staggered layout has (n[axis]+1) values in the normal direction and
// (n[ax]+2*ghost) values in each tangential direction:
//
//   storedDims[axis]      = n[axis] + 1         (no ghost in normal direction)
//   storedDims[other ax]  = n[other] + 2*ghost   (ghost padding, same as cell field)
//
// Face index (no ghost offset in normal direction):
//   Linear index for face (i_n, j_t0, k_t1):
//     flat = i_n + storedDims[axis] * (j_t0+ghost + storedDims[t0] * (k_t1+ghost))
//
// This is encoded in FieldLayout::index() for FACE_X/Y/Z centering.
//
// Physical face range along normal axis: i_n in [0, n[axis]]  (inclusive,
// because there are n+1 faces for n cells).
//
// Coordinate of face i_n along normal axis:
//   x_face = mesh.origin[axis] + i_n * mesh.d[axis]   (NOT +0.5*dx)
//
// CPU arrays are always valid after construction.
// GPU arrays are allocated lazily via allocDevice().
// ---------------------------------------------------------------------------

class FaceField {
public:
    // -----------------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------------
    std::string  name;
    FieldLayout  layout;
    int          normalAxis;  // 0=x, 1=y, 2=z

    // -----------------------------------------------------------------------
    // Geometry (cached from Mesh at construction)
    // -----------------------------------------------------------------------
    const Mesh& mesh;
    int         ghost;
    int         storedDims[3];
    std::size_t storedSize;

    // -----------------------------------------------------------------------
    // CPU storage
    // -----------------------------------------------------------------------
    std::vector<Real> data;  // current face values

    // -----------------------------------------------------------------------
    // GPU storage (nullptr until allocDevice())
    // -----------------------------------------------------------------------
    Real* d_data = nullptr;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    explicit FaceField(const Mesh& mesh,
                       int normalAxis,
                       const std::string& name,
                       int ghost = 1);

    // Non-copyable (owns GPU memory)
    FaceField(const FaceField&)            = delete;
    FaceField& operator=(const FaceField&) = delete;

    // Movable
    FaceField(FaceField&& other) noexcept;
    FaceField& operator=(FaceField&& other) noexcept;

    ~FaceField();

    // -----------------------------------------------------------------------
    // Index mapping
    // i_n: normal index [0, n[axis]]   (n+1 faces)
    // i_t0, i_t1: tangential physical cell indices (ghost-padded in layout)
    // -----------------------------------------------------------------------
    int index(int i, int j, int k) const { return layout.index(i, j, k); }
    int index(int i, int j)        const { return layout.index(i, j, 0); }
    int index(int i)               const { return layout.index(i, 0, 0); }

    // Physical coordinate of face i_n along normalAxis
    double faceCoord(int i_n) const {
        return mesh.origin[normalAxis] + i_n * mesh.d[normalAxis];
    }

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------
    void fill(double value);

    // Initialize by evaluating fn(x, y, z) at each face position.
    // The coordinate along normalAxis is the face position (no +0.5*d);
    // the other two coordinates are cell-centre positions.
    // fn signature: double fn(double x, double y, double z)
    template<typename Fn>
    void initialize(Fn fn) {
        const int ax  = normalAxis;
        const int nx  = (ax == 0) ? mesh.n[0] + 1 : mesh.n[0];
        const int ny  = (ax == 1) ? mesh.n[1] + 1 : mesh.n[1];
        const int nz  = (ax == 2) ? mesh.n[2] + 1 : mesh.n[2];

        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            double x = (ax == 0) ? (mesh.origin[0] + i * mesh.d[0])
                                 : mesh.coord(0, i);
            double y = (ax == 1) ? (mesh.origin[1] + j * mesh.d[1])
                                 : ((mesh.dim >= 2) ? mesh.coord(1, j) : 0.0);
            double z = (ax == 2) ? (mesh.origin[2] + k * mesh.d[2])
                                 : ((mesh.dim >= 3) ? mesh.coord(2, k) : 0.0);
            data[static_cast<std::size_t>(layout.index(i, j, k))] = fn(x, y, z);
        }
    }

    // -----------------------------------------------------------------------
    // GPU management
    // -----------------------------------------------------------------------
    bool deviceAllocated() const { return d_data != nullptr; }

    void allocDevice();
    void freeDevice();

    void uploadToDevice()   const;  // CPU data -> GPU d_data
    void downloadFromDevice();      // GPU d_data -> CPU data

private:
    bool ownsDeviceMemory_ = true;
};

} // namespace PhiX
