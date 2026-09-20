#pragma once

// ---------------------------------------------------------------------------
// MovingFrame.h — moving computational window ("pulled frame") for fronts
// that outgrow the domain: directional solidification, travelling waves,
// any "follow the interface" problem.
//
// The window advances one cell along `axis` toward `advance` per shift():
// every registered field's content moves one cell the other way, the slice
// at the trailing side exits (optionally recorded, see conveyor()), and the
// slice at the leading side is refilled by that field's injection rule.
//
//     MovingFrame frame(mesh, Axis::X, Side::HIGH);
//     frame.add(psi, InjectRule::slope(-dxW), /*record=*/true);
//     frame.add(U,   InjectRule::constant(-1.0));
//     frame.add(p,   InjectRule::constant(0.0), /*record=*/true);
//     frame.add(thT, InjectRule::slope(+dxW / lT));
//     ...
//     if (step % shiftCheck == 0)
//         frame.trackTo(x0W + dxW, [&] {           // keep front near x0W
//             return reduce::fieldMaxPW(psi, xc, PHIX_FN (Real p, Real x) {
//                 return p > Real(0) ? x : Real(0); });
//         });
//
// Correctness invariants:
//
//   • Injection base cell.  InjectRule::slope continues from THE LEADING
//     CELL'S OWN pre-shift value: new[last] = old[last] + delta.
//     Extrapolating from old[last-1] is wrong: after the shift
//     new[last-1] = old[last], so a linear field gets
//     new[last] - new[last-1] = 0 — every shift injects a flat pair and
//     the pulled gradient decays (self-excited front runaway).
//
//   • shift() stages every field through an internal device scratch
//     buffer; in-place shifting is a data race.
//
//   • trackTo() throws NumericsError after shifting a whole domain
//     length in one call — that is a runaway, not front tracking.
//
// Requirements/limits: 2D meshes (or 1D along X); fields share the frame
// mesh and ghost layout, device-resident.  shift() touches PHYSICAL cells
// only — ghosts are stale afterwards; refresh BCs before the next stencil
// evaluation (the usual apply-BCs-at-loop-top pattern already does).
// Recording downloads only the exiting slice (strided copy), not the field.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "mesh/Mesh.h"
#include "mesh/Patch.h"

#include <functional>
#include <string>
#include <vector>

namespace PhiX {

// Outflow refill rule for one field (evaluated on the leading slice).
struct InjectRule {
    enum class Kind { Constant, Slope };
    Kind   kind;
    double value;   // Constant: the far-field value; Slope: the per-cell
                    // increment added to the leading cell's own old value

    static InjectRule constant(double v) { return {Kind::Constant, v}; }
    static InjectRule slope(double d)    { return {Kind::Slope,    d}; }
};

class MovingFrame {
public:
    // Window advances toward `advance` along `axis` (default: +X — the
    // classic pulled directional-solidification frame).
    explicit MovingFrame(const Mesh& mesh, Axis axis = Axis::X,
                         Side advance = Side::HIGH);
    ~MovingFrame();
    MovingFrame(const MovingFrame&) = delete;
    MovingFrame& operator=(const MovingFrame&) = delete;

    // Register a field (same mesh + ghost as the frame, device-resident by
    // shift time).  record = true keeps every exiting slice — see
    // conveyor().
    void add(ScalarField& f, InjectRule rule, bool record = false);

    // Shift the window one cell.  Race-free (staged through scratch);
    // physical cells only, ghosts left stale.
    void shift();

    // Shift while position() > anchor, re-evaluating after every shift.
    // Returns the number of shifts performed.  Throws NumericsError once
    // maxShiftsPerCall (default: one full domain length) is exceeded —
    // the runaway guard.
    int trackTo(double anchor, const std::function<double()>& position,
                int maxShiftsPerCall = -1);

    long shifts() const { return shifts_; }

    // Exited slices of a recorded field, oldest first, nAcross values per
    // shift (for Axis::X: ny values per column, j fastest).  Together with
    // the live window this reconstructs the full pulled sample.
    const std::vector<Real>& conveyor(const ScalarField& f) const;

private:
    struct Entry {
        ScalarField* field;
        InjectRule   rule;
        bool         record;
        std::vector<Real> conveyor;
    };

    const Mesh& mesh_;
    Axis  axis_;
    Side  advance_;
    int   nAlong_ = 0, nAcross_ = 0;
    long  strideAlong_ = 0, strideAcross_ = 0, base_ = 0;
    long  shifts_ = 0;

    std::vector<Entry> entries_;
    Real*       d_scratch_ = nullptr;   // one field's stored array
    std::size_t scratchBytes_ = 0;
    std::vector<Real> hostSlice_;     // nAcross staging for recording

    void ensureLayout(const ScalarField& f, const char* fn) const;
};

} // namespace PhiX
