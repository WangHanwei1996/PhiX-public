#pragma once

// ---------------------------------------------------------------------------
// Reduce.h — device-side reductions over the PHYSICAL cells of a ScalarField.
//
// All functions read the field's d_curr buffer and exclude the ghost halo
// (ghost cells may hold stale or poisoned values without affecting results).
// The field must be device-allocated (allocDevice + upload) beforehand.
//
// Every call is synchronous on the default CUDA stream: it blocks until the
// scalar result is back on the host.  Internally a small device scratch
// buffer is cached and reused across calls (grow-only); call freeScratch()
// to release it explicitly (e.g. before cudaDeviceReset).
//
// Typical uses: adaptive-dt control (fieldMaxAbs of an RHS field), NaN
// guards (fieldHasNonFinite), conservation checks (fieldSum), steady-state
// detection, on-GPU diagnostics without downloading whole fields.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"

#include <cuda_runtime.h>

namespace PhiX {
namespace reduce {

double fieldMax   (const ScalarField& f);   // max  over physical cells
double fieldMin   (const ScalarField& f);   // min  over physical cells
double fieldMaxAbs(const ScalarField& f);   // max |value|
double fieldSum   (const ScalarField& f);   // plain sum (multiply by cell
                                            //   volume for an integral)
double fieldSumSq (const ScalarField& f);   // sum of squares
double fieldL2    (const ScalarField& f);   // sqrt(fieldSumSq)

// True if any physical cell holds NaN or ±Inf.  Ghost cells are ignored.
bool fieldHasNonFinite(const ScalarField& f);

// Inner product Σ a·b over physical cells (double accumulation).
// Both fields must share the same layout.
double fieldDot(const ScalarField& a, const ScalarField& b);

// Asynchronous variant: the double result is written to the DEVICE slot
// d_out (no host copy, no synchronisation — enqueued on the default
// stream).  Building block for device-resident solver control flow (CG).
void fieldDotAsync(const ScalarField& a, const ScalarField& b, double* d_out,
                   cudaStream_t stream = nullptr);

// Σ over physical cells of |∇f|² (CD2 central differences — f's ghosts
// must be refreshed).  Multiply by κ/2·dV for the gradient energy term of
// a free-energy functional.  Double accumulation.
double fieldGradSq(const ScalarField& f);

// Release the internally cached device scratch buffers.
void freeScratch();

// Internal accessors for the header-only pointwise reductions
// (field/ReducePW.h) — cached device scratch shared with this module.
namespace detail {
void*   scratchTemp(std::size_t bytes);   // grow-only CUB temp storage
double* scratchOut();                     // 8-byte device result slot
}

} // namespace reduce
} // namespace PhiX
