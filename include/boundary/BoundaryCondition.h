#pragma once

#include "mesh/Patch.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// BoundaryCondition  --  abstract base class
//
// Bound to a Patch (a named slice of one mesh face).  The Patch determines
// which boundary cells this BC fills; the BC subclass determines HOW.
//
// Concrete subclasses implement applyOnCPU / applyOnGPU for ScalarField.
// VectorField overloads have default implementations that apply the BC
// independently to every component.  Override the vector overloads in a
// subclass when components must be coupled (e.g. slip boundaries).
// ---------------------------------------------------------------------------

class BoundaryCondition {
public:
    const Patch& patch;

    explicit BoundaryCondition(const Patch& p) : patch(p) {}
    virtual ~BoundaryCondition() = default;

    // Convenience: axis / side of the bound patch
    Axis axis() const { return patch.axis; }
    Side side() const { return patch.side; }

    // --- ScalarField interface (must be implemented by subclasses) ----------
    virtual void applyOnCPU(ScalarField& f) const = 0;
    virtual void applyOnGPU(ScalarField& f) const = 0;

    // --- VectorField interface (default: apply to each component) -----------
    virtual void applyOnCPU(VectorField& vf) const {
        for (int c = 0; c < vf.nComponents(); ++c) applyOnCPU(vf[c]);
    }
    virtual void applyOnGPU(VectorField& vf) const {
        for (int c = 0; c < vf.nComponents(); ++c) applyOnGPU(vf[c]);
    }
};

} // namespace PhiX
