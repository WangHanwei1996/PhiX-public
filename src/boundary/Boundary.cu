#include "boundary/PeriodicBC.h"
#include "core/CudaCheck.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"
#include "boundary/BCBatch.h"
#include "mesh/Mesh.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

// ===========================================================================
// Generic patch-aware kernel design
// ===========================================================================
//
// Row-major storage: flat(is, js, ks) = is + sx*(js + sy*ks)
// where is/js/ks are stored indices (physical index + ghost offset).
//
// For each Patch (axis, side, IndexBox region) we identify:
//
//   axis (normal)         : the BC axis
//   tangential axes t0,t1 : the other two axes (t0 < t1)
//   axis_stride           : flat stride along the normal axis
//   t0_stride, t1_stride  : flat strides along the two tangential axes
//   t0_count, t1_count    : number of cells along each tangential axis,
//                           taken from region.extent(t0/t1)
//   t0_lo, t1_lo          : physical starting index along t0/t1, plus ghost
//
// Each thread handles one (s0, s1) cell within the patch's region.
// ===========================================================================

struct PatchParams {
    int axis_stride;
    int n_axis;        // physical cells along normal axis
    int ghost;
    int t0_stride;
    int t1_stride;
    int t0_count;
    int t1_count;
    int t0_lo;
    int t1_lo;
};

static PatchParams makePatchParams(const ScalarField& f, const Patch& p) {
    const int sx = f.storedDims[0];
    const int sy = f.storedDims[1];

    const int a = static_cast<int>(p.axis);
    int t0, t1;
    switch (a) {
        case 0: t0 = 1; t1 = 2; break;
        case 1: t0 = 0; t1 = 2; break;
        default: t0 = 0; t1 = 1; break;
    }

    auto axStride = [&](int axis) {
        if (axis == 0) return 1;
        if (axis == 1) return sx;
        return sx * sy;
    };

    PatchParams pp{};
    pp.axis_stride = axStride(a);
    pp.n_axis      = f.mesh.n[a];
    pp.ghost       = f.ghost;
    pp.t0_stride   = axStride(t0);
    pp.t1_stride   = axStride(t1);
    pp.t0_count    = p.region.extent(t0);
    pp.t1_count    = p.region.extent(t1);
    pp.t0_lo       = p.region.lo[t0] + f.ghost;
    pp.t1_lo       = p.region.lo[t1] + f.ghost;
    return pp;
}

// ---------------------------------------------------------------------------
// Periodic kernel (single patch handles BOTH sides of its axis)
// ---------------------------------------------------------------------------
__global__ void kernel_periodic(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    for (int g = 1; g <= ghost; ++g) {
        int lo_ghost  = (ghost - g)              * axis_stride + face_off;
        int lo_source = (ghost + n_axis - g)     * axis_stride + face_off;
        int hi_ghost  = (ghost + n_axis + g - 1) * axis_stride + face_off;
        int hi_source = (ghost + g - 1)          * axis_stride + face_off;

        data[lo_ghost] = data[lo_source];
        data[hi_ghost] = data[hi_source];
    }
}

// ---------------------------------------------------------------------------
// NoFlux (zero-gradient) kernel — single side determined by `is_low`
// ---------------------------------------------------------------------------
__global__ void kernel_noflux(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost,
        bool is_low, bool reflect)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    if (is_low) {
        int src = ghost * axis_stride + face_off;
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = data[src + (reflect ? g - 1 : 0) * axis_stride];
        }
    } else {
        int src = (ghost + n_axis - 1) * axis_stride + face_off;
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = data[src - (reflect ? g - 1 : 0) * axis_stride];
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed (Dirichlet) kernel — single side determined by `is_low`
// ---------------------------------------------------------------------------
__global__ void kernel_fixed(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost,
        bool is_low,
        double value)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    if (is_low) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = value;
        }
    } else {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = value;
        }
    }
}

// ===========================================================================
// CPU helpers (mirror of device kernels)
// ===========================================================================

static void cpu_periodic(Real* data, const PatchParams& pp) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        for (int g = 1; g <= pp.ghost; ++g) {
            int lo_ghost  = (pp.ghost - g)                 * pp.axis_stride + face_off;
            int lo_source = (pp.ghost + pp.n_axis - g)     * pp.axis_stride + face_off;
            int hi_ghost  = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
            int hi_source = (pp.ghost + g - 1)             * pp.axis_stride + face_off;
            data[lo_ghost] = data[lo_source];
            data[hi_ghost] = data[hi_source];
        }
    }
}

static void cpu_noflux(Real* data, const PatchParams& pp, bool is_low, bool reflect) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        if (is_low) {
            int src = pp.ghost * pp.axis_stride + face_off;
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost - g) * pp.axis_stride + face_off;
                data[dst] = data[src + (reflect ? g - 1 : 0) * pp.axis_stride];
            }
        } else {
            int src = (pp.ghost + pp.n_axis - 1) * pp.axis_stride + face_off;
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
                data[dst] = data[src - (reflect ? g - 1 : 0) * pp.axis_stride];
            }
        }
    }
}

static void cpu_fixed(Real* data, const PatchParams& pp,
                      bool is_low, double value) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        if (is_low) {
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost - g) * pp.axis_stride + face_off;
                data[dst] = value;
            }
        } else {
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
                data[dst] = value;
            }
        }
    }
}

// ===========================================================================
// PeriodicBC
// ===========================================================================

PeriodicBC::PeriodicBC(const Patch& patch) : BoundaryCondition(patch) {}

void PeriodicBC::applyOnCPU(ScalarField& f) const {
    auto pp = makePatchParams(f, patch);
    cpu_periodic(f.curr.data(), pp);
}

void PeriodicBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("PeriodicBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_periodic<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// NoFluxBC
// ===========================================================================

NoFluxBC::NoFluxBC(const Patch& patch, Closure closure)
    : BoundaryCondition(patch), closure_(closure) {}

static void checkReflectedHalo(const ScalarField& f, const NoFluxBC& bc) {
    if (bc.closure() == NoFluxBC::Closure::Reflect &&
        (f.mesh.dim != 2 || f.ghost > f.mesh.n[int(bc.axis())]))
        throw std::invalid_argument("reflected NoFlux requires 2D and halo no larger than the domain");
}

void NoFluxBC::applyOnCPU(ScalarField& f) const {
    checkReflectedHalo(f, *this);
    auto pp = makePatchParams(f, patch);
    cpu_noflux(f.curr.data(), pp, patch.side == Side::LOW, closure_ == Closure::Reflect);
}

void NoFluxBC::applyOnGPU(ScalarField& f) const {
    checkReflectedHalo(f, *this);
    if (!f.deviceAllocated())
        throw std::runtime_error("NoFluxBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_noflux<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost,
        patch.side == Side::LOW, closure_ == Closure::Reflect);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// FixedBC
// ===========================================================================

FixedBC::FixedBC(const Patch& patch, double value)
    : BoundaryCondition(patch), value(value) {}

void FixedBC::applyOnCPU(ScalarField& f) const {
    auto pp = makePatchParams(f, patch);
    cpu_fixed(f.curr.data(), pp, patch.side == Side::LOW, value);
}

void FixedBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("FixedBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_fixed<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost,
        patch.side == Side::LOW,
        value);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// BCBatch — all descriptors of one field handled by a single kernel launch.
// The per-type bodies mirror kernel_periodic / kernel_noflux / kernel_fixed
// exactly; descriptors write disjoint ghost bands and read only physical
// cells, so concurrent execution inside one launch is hazard-free.
// ===========================================================================

__global__ void kernel_bc_batched(Real* data, const BCDesc* descs, int pass)
{
    const BCDesc d = descs[blockIdx.z];

    // pass 1 = corner fill: only descriptors with an extension run, over
    // tangential ranges widened into the (already filled) ghost bands.
    int lo0 = d.t0_lo, n0 = d.t0_count;
    int lo1 = d.t1_lo, n1 = d.t1_count;
    if (pass == 1) {
        if (!d.ext0 && !d.ext1) return;
        if (d.ext0) { lo0 -= d.ghost; n0 += 2 * d.ghost; }
        if (d.ext1) { lo1 -= d.ghost; n1 += 2 * d.ghost; }
    }

    const int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= n0 || s1 >= n1) return;

    const int face_off = (lo0 + s0) * d.t0_stride
                       + (lo1 + s1) * d.t1_stride;

    switch (d.type) {
    case 0:   // periodic — one descriptor fills BOTH sides of its axis
        for (int g = 1; g <= d.ghost; ++g) {
            const int lo_ghost  = (d.ghost - g)              * d.axis_stride + face_off;
            const int lo_source = (d.ghost + d.n_axis - g)   * d.axis_stride + face_off;
            const int hi_ghost  = (d.ghost + d.n_axis + g - 1) * d.axis_stride + face_off;
            const int hi_source = (d.ghost + g - 1)          * d.axis_stride + face_off;
            data[lo_ghost] = data[lo_source];
            data[hi_ghost] = data[hi_source];
        }
        break;
    case 1:   // no-flux (constant extension)
    case 3:   // no-flux (even reflection)
        if (d.is_low) {
            const int src = d.ghost * d.axis_stride + face_off;
            for (int g = 1; g <= d.ghost; ++g)
                data[(d.ghost - g) * d.axis_stride + face_off] =
                    data[src + (d.type == 3 ? g - 1 : 0) * d.axis_stride];
        } else {
            const int src = (d.ghost + d.n_axis - 1) * d.axis_stride + face_off;
            for (int g = 1; g <= d.ghost; ++g)
                data[(d.ghost + d.n_axis + g - 1) * d.axis_stride + face_off] =
                    data[src - (d.type == 3 ? g - 1 : 0) * d.axis_stride];
        }
        break;
    default:  // fixed (Dirichlet)
        if (d.is_low) {
            for (int g = 1; g <= d.ghost; ++g)
                data[(d.ghost - g) * d.axis_stride + face_off] = d.value;
        } else {
            for (int g = 1; g <= d.ghost; ++g)
                data[(d.ghost + d.n_axis + g - 1) * d.axis_stride + face_off] = d.value;
        }
        break;
    }
}

BCBatch::~BCBatch() {
    if (d_descs_) cudaFree(d_descs_);   // best-effort; no throw in dtor
}

void BCBatch::build(const ScalarField& layoutRef,
                    const std::vector<BoundaryCondition*>& bcs)
{
    std::vector<BCDesc> descs;
    fallback_.clear();

    for (BoundaryCondition* bc : bcs) {
        if (!bc) continue;
        // The BC's patch must be one of THIS mesh's patches (BCs hold a
        // reference into mesh.patches()); a foreign or dangling patch would
        // write ghost cells with a mismatched geometry.  Note addPatch()
        // may reallocate the patch vector — build BCs after all patch edits.
        {
            bool owned = false;
            for (const auto& p : layoutRef.mesh.patches())
                if (&p == &bc->patch) { owned = true; break; }
            if (!owned)
                throw ValidationError("BCBatch::build",
                    "BC patch '" + bc->patch.name
                    + "' does not belong to the mesh of field '"
                    + layoutRef.name + "'",
                    "construct BCs from patches of the SAME Mesh object the "
                    "field was built on (mesh.facePatch/patch), after all "
                    "patch edits are done");
        }
        const PatchParams pp = makePatchParams(layoutRef, bc->patch);
        BCDesc d{};
        d.is_low      = (bc->patch.side == Side::LOW) ? 1 : 0;
        d.t0_count    = pp.t0_count;    d.t1_count = pp.t1_count;
        d.t0_stride   = pp.t0_stride;   d.t1_stride = pp.t1_stride;
        d.t0_lo       = pp.t0_lo;       d.t1_lo = pp.t1_lo;
        d.axis_stride = pp.axis_stride; d.n_axis = pp.n_axis;
        d.ghost       = pp.ghost;
        d.value       = Real(0);

        // Corner-pass extension: tangential axes with GLOBAL index greater
        // than this BC's axis (and active in the mesh) get extended in
        // pass 1 — write regions stay disjoint across descriptors.
        {
            const int a = static_cast<int>(bc->patch.axis);
            int t0a, t1a;
            switch (a) {
                case 0:  t0a = 1; t1a = 2; break;
                case 1:  t0a = 0; t1a = 2; break;
                default: t0a = 0; t1a = 1; break;
            }
            const int dim = layoutRef.mesh.dim;
            d.ext0 = (t0a > a && t0a < dim) ? 1 : 0;
            d.ext1 = (t1a > a && t1a < dim) ? 1 : 0;
        }

        if (dynamic_cast<const PeriodicBC*>(bc)) {
            d.type = 0;
        } else if (const auto* nf = dynamic_cast<const NoFluxBC*>(bc)) {
            checkReflectedHalo(layoutRef, *nf);
            d.type = nf->closure() == NoFluxBC::Closure::Reflect ? 3 : 1;
        } else if (const auto* fx = dynamic_cast<const FixedBC*>(bc)) {
            d.type  = 2;
            d.value = static_cast<Real>(fx->value);
        } else {
            fallback_.push_back(bc);   // unknown subclass: keep old path
            continue;
        }
        descs.push_back(d);
    }

    if (d_descs_) { cudaFree(d_descs_); d_descs_ = nullptr; }
    nDesc_ = static_cast<int>(descs.size());
    maxT0_ = maxT1_ = 0;
    for (const auto& d : descs) {
        maxT0_ = (d.t0_count > maxT0_) ? d.t0_count : maxT0_;
        maxT1_ = (d.t1_count > maxT1_) ? d.t1_count : maxT1_;
    }
    // grid must cover the extended (corner-pass) ranges as well
    hasExt_ = false;
    for (const auto& d : descs) {
        if (d.ext0) {
            hasExt_ = true;
            maxT0_ = (d.t0_count + 2 * d.ghost > maxT0_)
                         ? d.t0_count + 2 * d.ghost : maxT0_;
        }
        if (d.ext1) {
            hasExt_ = true;
            maxT1_ = (d.t1_count + 2 * d.ghost > maxT1_)
                         ? d.t1_count + 2 * d.ghost : maxT1_;
        }
    }
    if (nDesc_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_descs_, nDesc_ * sizeof(BCDesc)));
        CUDA_CHECK(cudaMemcpy(d_descs_, descs.data(),
                              nDesc_ * sizeof(BCDesc),
                              cudaMemcpyHostToDevice));
    }
    built_ = true;
}

void BCBatch::applyOnGPU(ScalarField& f, cudaStream_t stream) const
{
    if (!built_)
        throw std::runtime_error("BCBatch::applyOnGPU: build() not called");
    if (nDesc_ > 0) {
        if (!f.d_curr)
            throw std::runtime_error("BCBatch::applyOnGPU: field not on device");
        dim3 block(16, 16);
        dim3 grid((maxT0_ + 15) / 16, (maxT1_ + 15) / 16, nDesc_);
        kernel_bc_batched<<<grid, block, 0, stream>>>(f.d_curr, d_descs_, 0);
        CUDA_CHECK(cudaGetLastError());
        if (hasExt_) {   // corner pass (reads pass-0 ghost bands)
            kernel_bc_batched<<<grid, block, 0, stream>>>(f.d_curr,
                                                          d_descs_, 1);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    for (BoundaryCondition* bc : fallback_) bc->applyOnGPU(f);
}

} // namespace PhiX
