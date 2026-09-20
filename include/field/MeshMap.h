#pragma once

// ---------------------------------------------------------------------------
// MeshMap.h — field transfer between two uniform Cartesian meshes covering
// the SAME physical domain (arbitrary resolutions/origins per axis; no
// integer-ratio or node-alignment requirement).
//
// This is the binding layer for multi-mesh solvers: each field keeps its
// own mesh (the framework is already per-equation-mesh throughout — scratch,
// BCs and health sentinels all follow the unknown's mesh), and a MeshMap
// carries values across explicitly.  Mixing meshes inside one Term/Expr
// remains rejected by the check layer — route every crossing through a map:
//
//   Mesh coarse = Mesh::makeUniform2D(...);   // e.g. temperature grid
//   Mesh fine   = Mesh::makeUniform2D(...);   // e.g. phase-field grid
//   MeshMap map(coarse, fine);                // validates same domain
//
//   map.interpolate(T_coarse, T_on_fine);           // non-conservative
//   map.conserve   (c_fine,   c_on_coarse);         // ∫c dV preserved
//   map.interpolate(T_coarse, T_on_fine, bcsFine);  // + ghost refresh
//
// Semantics:
//   interpolate — bi/tri-linear sampling of the source at the destination
//     cell centres.  Reads physical source cells only (edge-clamped in the
//     boundary half-cell), so it does NOT require fresh source ghosts.
//     Right choice for non-conserved quantities (T, driving forces, φ).
//   conserve — exact cell-overlap volume weighting between the two uniform
//     grids: Σ f·dV is preserved to roundoff in either direction.  Right
//     choice for conserved quantities (composition, energy density).
//     For integer-ratio aligned grids the coarse←fine direction reduces to
//     the exact block average.
//
// Direction is inferred per call from the fields' meshes (each argument
// must live on one of the two constructor meshes).  Both directions are
// supported by both operators.  Destination ghosts are stale after a plain
// transfer — use the bcs overloads or apply BCs yourself before stencils.
// Fields must be device-allocated; transfers run on the given stream.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "mesh/Mesh.h"

#include <cuda_runtime.h>

#include <vector>

namespace PhiX {

class BoundaryCondition;

class MeshMap {
public:
    // Validates: both meshes Cartesian, same dim, same physical extent and
    // origin per active axis (relative tolerance 1e-12).  Throws
    // ValidationError otherwise.
    MeshMap(const Mesh& a, const Mesh& b);

    const Mesh& meshA() const { return a_; }
    const Mesh& meshB() const { return b_; }

    // True when the two grids are integer-ratio nested with aligned cell
    // boundaries on every active axis (informational; the general kernels
    // are exact for this case too).
    bool aligned() const { return aligned_; }

    // --- transfers (direction inferred from the fields' meshes) -----------
    void interpolate(const ScalarField& src, ScalarField& dst,
                     cudaStream_t stream = nullptr) const;
    void conserve   (const ScalarField& src, ScalarField& dst,
                     cudaStream_t stream = nullptr) const;

    // Transfer + refresh the destination field's ghosts with its own BCs.
    void interpolate(const ScalarField& src, ScalarField& dst,
                     const std::vector<BoundaryCondition*>& dstBCs,
                     cudaStream_t stream = nullptr) const;
    void conserve   (const ScalarField& src, ScalarField& dst,
                     const std::vector<BoundaryCondition*>& dstBCs,
                     cudaStream_t stream = nullptr) const;

    // Component-wise convenience for VectorFields (same component count).
    void interpolate(const VectorField& src, VectorField& dst,
                     cudaStream_t stream = nullptr) const;
    void conserve   (const VectorField& src, VectorField& dst,
                     cudaStream_t stream = nullptr) const;

private:
    // Checks src/dst live on {a_, b_} in either order; throws otherwise.
    void checkPair(const ScalarField& src, const ScalarField& dst,
                   const char* who) const;

    const Mesh& a_;
    const Mesh& b_;
    bool aligned_ = false;
};

} // namespace PhiX
