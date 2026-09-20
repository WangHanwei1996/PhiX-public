#include "solver/MovingFrame.h"

#include "core/CudaCheck.h"
#include "core/Error.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// One field, one cell of window advance.  dir = +1: content moves one cell
// toward LOW (window advances toward HIGH), the slice at along = nAlong-1
// is refilled by the rule; dir = -1 mirrored.  kind 0 = Constant, 1 = Slope
// (v added to the leading cell's OWN pre-shift value — the correct
// extrapolation base, see the header's flat-pair trap note).
__global__ void k_frame_shift(const Real* src, Real* dst,
                              int nAlong, int nAcross,
                              long strideAlong, long strideAcross, long base,
                              int dir, int kind, Real v)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= nAlong || b >= nAcross) return;
    const long c = base + (long)a * strideAlong + (long)b * strideAcross;
    if (dir > 0) {
        if (a < nAlong - 1) dst[c] = src[c + strideAlong];
        else                dst[c] = (kind == 0) ? v : src[c] + v;
    } else {
        if (a > 0) dst[c] = src[c - strideAlong];
        else       dst[c] = (kind == 0) ? v : src[c] + v;
    }
}

} // namespace

MovingFrame::MovingFrame(const Mesh& mesh, Axis axis, Side advance)
    : mesh_(mesh), axis_(axis), advance_(advance)
{
    if (mesh.dim > 2)
        throw std::invalid_argument(
            "MovingFrame: 1D/2D meshes only for now (3D slice recording "
            "is not implemented)");
    const int ax = static_cast<int>(axis);
    if (ax >= mesh.dim)
        throw std::invalid_argument(
            "MovingFrame: axis exceeds mesh dimensionality");
    nAlong_  = mesh.n[ax];
    nAcross_ = (mesh.dim == 2) ? mesh.n[1 - ax] : 1;
    if (nAlong_ < 2)
        throw std::invalid_argument(
            "MovingFrame: need at least 2 cells along the frame axis");
}

MovingFrame::~MovingFrame() {
    if (d_scratch_) cudaFree(d_scratch_);
}

void MovingFrame::ensureLayout(const ScalarField& f, const char* fn) const {
    if (&f.mesh != &mesh_ && (f.mesh.n[0] != mesh_.n[0]
                              || f.mesh.n[1] != mesh_.n[1]
                              || f.mesh.n[2] != mesh_.n[2]))
        throw std::invalid_argument(
            std::string(fn) + ": field '" + f.name
            + "' is not on the frame's mesh");
    if (!entries_.empty()) {
        const ScalarField& ref = *entries_.front().field;
        if (f.storedSize != ref.storedSize || f.ghost != ref.ghost)
            throw std::invalid_argument(
                std::string(fn) + ": field '" + f.name
                + "' layout differs from '" + ref.name + "'");
    }
}

void MovingFrame::add(ScalarField& f, InjectRule rule, bool record) {
    ensureLayout(f, "MovingFrame::add");
    for (const Entry& e : entries_)
        if (e.field == &f)
            throw std::invalid_argument(
                "MovingFrame::add: field '" + f.name + "' added twice");

    if (entries_.empty()) {
        // layout constants from the first field (they all share it)
        const int g  = f.ghost;
        const long sx = f.storedDims[0];
        const long sy = f.storedDims[1];
        base_ = g + sx * (g + sy * g);
        const int ax = static_cast<int>(axis_);
        strideAlong_  = (ax == 0) ? 1 : sx;
        strideAcross_ = (ax == 0) ? sx : 1;
        hostSlice_.resize(static_cast<std::size_t>(nAcross_));
    }
    entries_.push_back(Entry{&f, rule, record, {}});
}

void MovingFrame::shift() {
    if (entries_.empty())
        throw std::invalid_argument("MovingFrame::shift: no fields added");

    ScalarField& ref = *entries_.front().field;
    if (!d_scratch_) {
        scratchBytes_ = ref.storedBytes();
        CUDA_CHECK(cudaMalloc(&d_scratch_, scratchBytes_));
    }

    const int  dir     = (advance_ == Side::HIGH) ? +1 : -1;
    const int  exitIdx = (dir > 0) ? 0 : nAlong_ - 1;
    const dim3 blk(32, 8);
    const dim3 grd((nAlong_ + 31) / 32, (nAcross_ + 7) / 8);

    for (Entry& e : entries_) {
        ScalarField& f = *e.field;
        if (!f.d_curr)
            throw std::runtime_error(
                "MovingFrame::shift: field '" + f.name
                + "' has no device allocation");

        // stage: race-free source copy (in-place shifting is a data race)
        CUDA_CHECK(cudaMemcpy(d_scratch_, f.d_curr, scratchBytes_,
                              cudaMemcpyDeviceToDevice));

        // record the exiting slice (strided device->host, slice only)
        if (e.record) {
            const Real* src = d_scratch_ + base_
                              + static_cast<long>(exitIdx) * strideAlong_;
            CUDA_CHECK(cudaMemcpy2D(hostSlice_.data(), sizeof(Real),
                                    src, strideAcross_ * sizeof(Real),
                                    sizeof(Real), nAcross_,
                                    cudaMemcpyDeviceToHost));
            e.conveyor.insert(e.conveyor.end(),
                              hostSlice_.begin(), hostSlice_.end());
        }

        k_frame_shift<<<grd, blk>>>(
            d_scratch_, f.d_curr, nAlong_, nAcross_,
            strideAlong_, strideAcross_, base_, dir,
            e.rule.kind == InjectRule::Kind::Constant ? 0 : 1,
            static_cast<Real>(e.rule.value));
        PHIX_KERNEL_CHECK("MovingFrame::shift");
    }
    ++shifts_;
}

int MovingFrame::trackTo(double anchor,
                         const std::function<double()>& position,
                         int maxShiftsPerCall) {
    if (maxShiftsPerCall < 0) maxShiftsPerCall = nAlong_;
    int n = 0;
    while (position() > anchor) {
        if (n >= maxShiftsPerCall)
            throw NumericsError("MovingFrame::trackTo",
                "shifted " + std::to_string(n) + " cells in one call and "
                "the position is still past the anchor",
                "a whole domain length per call is not front tracking — "
                "check the position callback and look for a physical "
                "runaway (flattened gradient, solute sink, wrong anchor)");
        shift();
        ++n;
    }
    return n;
}

const std::vector<Real>& MovingFrame::conveyor(const ScalarField& f) const {
    for (const Entry& e : entries_)
        if (e.field == &f) {
            if (!e.record)
                throw std::invalid_argument(
                    "MovingFrame::conveyor: field '" + f.name
                    + "' was added without record = true");
            return e.conveyor;
        }
    throw std::invalid_argument(
        "MovingFrame::conveyor: field '" + f.name + "' not added");
}

} // namespace PhiX
