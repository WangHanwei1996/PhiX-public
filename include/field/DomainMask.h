#pragma once

// ---------------------------------------------------------------------------
// DomainMask.h — non-rectangular computational domains on a structured grid
// (PFHub BM1c/BM2c T-shape etc.) via cell masking.
//
// A DomainMask is built once from an inside(x,y,z) predicate evaluated at
// cell centres.  Active cells carry mask 1, inactive cells (and all ghost
// cells) carry 0.  Two complementary mechanisms enforce a no-flux internal
// boundary at the mask edge:
//
//  - maskFaces(F): zeroes every face flux that is not between two ACTIVE
//    cells.  Used inside the conservative face-flux chain
//    (faceGrad → facePW → maskFaces → divFace); conservation over the
//    active region is then exact (telescoping sum), the gold standard.
//
//  - applyClosure(f): mirrors active-neighbour values into inactive cells
//    bordering the active region (average over active face-neighbours),
//    so a cell-based CD2 laplacian sees a zero normal gradient.  Call it
//    after each BC refresh, before stencil evaluation.  Supports
//    ghost-1 5/7-point stencils; for wider stencils or exact conservation
//    use the face-flux route.
//
// Masked reductions: sum(f) returns Σ f over active cells only; for other
// integrands pass cellMask() as an extra field to reduce::fieldSumPW.
// All operations are GPU-resident; the mask is uploaded at construction.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "mesh/Mesh.h"

#include <functional>

namespace PhiX {

class DomainMask {
public:
    // inside(x,y,z) → true for cells belonging to the computational domain
    DomainMask(const Mesh& mesh,
               const std::function<bool(double, double, double)>& inside,
               int ghost = 1);

    DomainMask(const DomainMask&)            = delete;
    DomainMask& operator=(const DomainMask&) = delete;

    // 1/0 cell mask (device-resident; ghosts are 0). Pass to
    // reduce::fieldSumPW for custom masked integrals.
    const ScalarField& cellMask() const { return mask_; }

    long long activeCells() const { return nActive_; }

    // zero every face not between two active cells (GPU, in place)
    void maskFaces(FaceField& F) const;

    // mirror closure: inactive cells bordering active ones receive the
    // average of their active face-neighbours (GPU, in place, d_curr)
    void applyClosure(ScalarField& f) const;

    // Σ f over active cells (ghosts/inactive cells never contribute)
    double sum(const ScalarField& f) const;

private:
    const Mesh& mesh_;
    ScalarField mask_;
    long long   nActive_ = 0;
};

} // namespace PhiX
