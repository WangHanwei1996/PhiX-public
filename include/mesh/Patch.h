#pragma once

#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Axis / Side enumerations
//
// (Defined here, shared by Mesh / Patch / BoundaryCondition.)
// ---------------------------------------------------------------------------
enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Side { LOW, HIGH };

// ---------------------------------------------------------------------------
// PatchKind
//
//   PHYSICAL  — a physical boundary (Dirichlet / Neumann / Periodic / ...).
//   PROCESSOR — reserved for future multi-GPU / MPI halo exchange patches.
//
// Only PHYSICAL is implemented in this version; PROCESSOR is recognised by
// the topology system but rejected by all BC application paths.
// ---------------------------------------------------------------------------
enum class PatchKind { PHYSICAL, PROCESSOR };

// ---------------------------------------------------------------------------
// IndexBox
//
// Half-open cell-index region [lo, hi) in 3D mesh coordinates.
//
//   - All indices refer to PHYSICAL cells (no ghost offset).
//   - For a Patch on (axis, side), the index range along `axis` is always
//     a single cell wide:
//         side == LOW   ->  lo[axis] = 0,            hi[axis] = 1
//         side == HIGH  ->  lo[axis] = mesh.n[axis]-1, hi[axis] = mesh.n[axis]
//     The other two (tangential) axes describe what fraction of the face
//     this patch covers.
// ---------------------------------------------------------------------------
struct IndexBox {
    int lo[3] = {0, 0, 0};
    int hi[3] = {0, 0, 0};

    int extent(int axis) const { return hi[axis] - lo[axis]; }

    long long numCells() const {
        return static_cast<long long>(extent(0)) * extent(1) * extent(2);
    }

    bool overlaps(const IndexBox& other) const {
        for (int a = 0; a < 3; ++a) {
            if (hi[a] <= other.lo[a] || other.hi[a] <= lo[a]) return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Patch
//
// A named slice of one face of the mesh.  Defined by:
//   - `name`   : unique identifier within a Mesh
//   - `axis`   : the face-normal axis (X / Y / Z)
//   - `side`   : LOW or HIGH end of that axis
//   - `region` : IndexBox describing which physical cells this patch covers
//   - `kind`   : PHYSICAL by default
//
// Sub-face patches: multiple Patches on the same (axis, side) may divide
// the face into non-overlapping regions; together they must cover the
// entire face.  Mesh::validatePatches() enforces this.
// ---------------------------------------------------------------------------
struct Patch {
    std::string name;
    Axis        axis;
    Side        side;
    IndexBox    region;
    PatchKind   kind = PatchKind::PHYSICAL;

    // Number of boundary cells covered by this patch.
    long long numCells() const { return region.numCells(); }
};

// ---------------------------------------------------------------------------
// Default face-patch naming convention used by Mesh::Mesh().
//   axis=X side=LOW  -> "xmin"      axis=X side=HIGH -> "xmax"
//   axis=Y side=LOW  -> "ymin"      axis=Y side=HIGH -> "ymax"
//   axis=Z side=LOW  -> "zmin"      axis=Z side=HIGH -> "zmax"
// ---------------------------------------------------------------------------
const char* defaultPatchName(Axis axis, Side side);

} // namespace PhiX
