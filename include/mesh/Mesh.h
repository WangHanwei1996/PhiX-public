#pragma once

#include "mesh/Patch.h"

#include <string>
#include <vector>
#include <stdexcept>

namespace PhiX {

enum class CoordSys {
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL
};

class Mesh {
public:
    // ---------------------------------------------------------------
    // Data members (all public for easy kernel-side read)
    // ---------------------------------------------------------------
    int      dim;
    CoordSys coordSys;
    int      n[3];       // nx, ny, nz  (unused directions set to 1)
    double   d[3];       // dx, dy, dz
    double   origin[3];  // x0, y0, z0

    // ---------------------------------------------------------------
    // Construction
    //
    // All public parameterised constructors call defaultPatches(): a
    // fresh Mesh always has 2*dim default full-face patches named
    // "xmin"/"xmax"/"ymin"/... covering each physical boundary.
    // ---------------------------------------------------------------

    // Default: uninitialized (call isValid() before use; has no patches)
    Mesh();

    // Direct parameter constructor
    Mesh(int dim, CoordSys cs,
         int nx, double dx, double x0,
         int ny, double dy, double y0,
         int nz, double dz, double z0);

    // Static factory helpers (cleaner call sites for 1D/2D)
    static Mesh makeUniform1D(CoordSys cs,
                              int nx, double dx, double x0 = 0.0);

    static Mesh makeUniform2D(CoordSys cs,
                              int nx, double dx, double x0,
                              int ny, double dy, double y0);

    static Mesh makeUniform3D(CoordSys cs,
                              int nx, double dx, double x0,
                              int ny, double dy, double y0,
                              int nz, double dz, double z0);

    // ---------------------------------------------------------------
    // IO
    // ---------------------------------------------------------------
    static Mesh readFromFile(const std::string& path);
    void write(const std::string& path) const;
    void print() const;

    // ---------------------------------------------------------------
    // Queries (all inline, safe to call from CPU kernel wrappers)
    // ---------------------------------------------------------------

    // Total number of grid points
    inline std::size_t totalSize() const {
        return static_cast<std::size_t>(n[0]) * n[1] * n[2];
    }

    // Cell-centre coordinate along axis (0=x,1=y,2=z) at index i
    inline double coord(int axis, int i) const {
        return origin[axis] + (i + 0.5) * d[axis];
    }

    // Row-major (x fast, z slow) linear index: index(i,j,k)
    inline int index(int i, int j, int k) const {
        return i + n[0] * (j + n[1] * k);
    }

    // 1-D convenience
    inline int index(int i) const { return i; }

    // 2-D convenience
    inline int index(int i, int j) const { return i + n[0] * j; }

    // Validate all parameters; throws std::invalid_argument on failure
    void validate() const;

    // Returns true / false without throwing
    bool isValid() const noexcept;

    // ===============================================================
    // Patch topology
    // ===============================================================

    // Read-only access to all patches.
    const std::vector<Patch>& patches() const { return patches_; }

    // Find patch by name; throws std::out_of_range if not found.
    const Patch&  patch(const std::string& name) const;
    Patch&        patch(const std::string& name);

    // True if a patch with this name exists.
    bool hasPatch(const std::string& name) const;

    // List all patches on (axis, side) in insertion order.
    std::vector<const Patch*> facePatches(Axis axis, Side side) const;

    // Convenience: return the (single) patch covering an entire face.
    // Throws std::runtime_error if the face has been split.
    const Patch& facePatch(Axis axis, Side side) const;

    // Add a patch.  Name must be unique.  Throws std::invalid_argument
    // on duplicate name or on a region inconsistent with (axis, side)
    // bounds.  Does NOT check topology completeness — call
    // validatePatches() at the end of all patch edits.
    void addPatch(const Patch& p);

    // Remove a single patch by name; returns true if found and removed.
    bool removePatch(const std::string& name);

    // Remove all patches on a given (axis, side).  Typical workflow:
    //   mesh.removeFacePatches(Axis::X, Side::LOW);
    //   mesh.addPatch({...});  mesh.addPatch({...});
    //   mesh.validatePatches();
    void removeFacePatches(Axis axis, Side side);

    // Verify that for every active (axis, side) the union of all patches
    // exactly tiles the face with no overlap.  Throws std::runtime_error
    // on any violation.  Called automatically inside defaultPatches();
    // callers must invoke it manually after editing patches.
    void validatePatches() const;

    // Construct the default 2*dim full-face patches.  Existing patches
    // (if any) are cleared first.
    void defaultPatches();

    // Build the IndexBox that covers an entire face on (axis, side).
    // Public so callers can construct sub-face patches relative to it.
    IndexBox fullFaceBox(Axis axis, Side side) const;

private:
    // Common initialisation used by constructors
    void init(int dim_, CoordSys cs_,
              int nx, double dx, double x0,
              int ny, double dy, double y0,
              int nz, double dz, double z0);

    // Validate that `box` is a legal region for a patch on (axis, side).
    void validatePatchBox(Axis axis, Side side, const IndexBox& box) const;

    std::vector<Patch> patches_;
};

// String helpers (used in IO)
std::string coordSysToString(CoordSys cs);
CoordSys    stringToCoordSys(const std::string& s);

} // namespace PhiX
