#pragma once

#include "mesh/Mesh.h"

#include <cstddef>
#include <stdexcept>

namespace PhiX {

enum class Centering {
    CELL,
    FACE_X,  // face-normal in x: nx+1 values, no ghost in x
    FACE_Y,  // face-normal in y: ny+1 values, no ghost in y
    FACE_Z   // face-normal in z: nz+1 values, no ghost in z
};

// Return the normal axis for a face centering (0/1/2), or -1 for CELL.
constexpr int faceAxis(Centering c) {
    return (c == Centering::FACE_X) ? 0
         : (c == Centering::FACE_Y) ? 1
         : (c == Centering::FACE_Z) ? 2
         : -1;
}

constexpr bool isFace(Centering c) { return faceAxis(c) >= 0; }

// ---------------------------------------------------------------------------
// FieldLayout
//
// Describes the padded storage geometry of a field on a Mesh.
// The current implementation supports only cell-centred storage, but the
// object exists so later versions can extend it to face/node layouts without
// re-embedding storage rules into ScalarField / VectorField.
// ---------------------------------------------------------------------------
class FieldLayout {
public:
    const Mesh*  mesh = nullptr;
    int          ghost = 0;
    Centering    centering = Centering::CELL;
    int          storedDims[3] = {0, 0, 0};
    std::size_t  storedSize = 0;

    FieldLayout() = default;
    explicit FieldLayout(const Mesh& mesh,
                         int ghost = 1,
                         Centering centering = Centering::CELL);

    const Mesh& meshRef() const {
        if (!mesh) throw std::runtime_error("FieldLayout: mesh is null");
        return *mesh;
    }

    // Face-field index (no ghost shift along normal axis).
    // For CELL centering all three shifts are +ghost.
    // For FACE_X the i-shift is 0; j/k still +ghost.
    // Provides a uniform call site for both cell and face layouts.
    int index(int i, int j, int k) const {
        int ax = faceAxis(centering);
        int si = (ax == 0) ? i : (i + ghost);
        int sj = (ax == 1) ? j : (j + ghost);
        int sk = (ax == 2) ? k : (k + ghost);
        return si + storedDims[0] * (sj + storedDims[1] * sk);
    }
    int index(int i, int j) const { return index(i, j, 0); }
    int index(int i) const { return index(i, 0, 0); }

    // Flat offset of the first element of the k=0 stored plane, i.e. the
    // number of elements occupied by the leading z-ghost planes:
    //   planeOffset() = storedDims[0] * storedDims[1] * ghost
    // For a 2D field this is the start of the SINGLE physical mid-plane —
    // raw device buffers shadowing a field must be allocated with the full
    // storedSize and addressed from this offset (see ScalarField docs).
    // (FACE_Z layouts have no ghost planes along z: offset 0.)
    std::size_t planeOffset() const {
        return (faceAxis(centering) == 2)
             ? 0
             : static_cast<std::size_t>(storedDims[0]) * storedDims[1] * ghost;
    }
};

} // namespace PhiX
