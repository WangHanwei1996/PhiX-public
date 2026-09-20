#include "field/GibbsSimplex.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Kernel
//
// d_fields[i] points to the d_curr buffer of the i-th phase field.
// The stored layout (with ghost cells) uses:
//   flat_idx = (ix+g) + sx*((iy+g) + sy*(iz+g))
//
// Algorithm (per cell):
//   1. Read all phi[i].
//   2. While any phi[i] < 0:
//        - sum the deficit of all negative entries
//        - zero them out
//        - distribute deficit equally among remaining positive entries
//      (iterate until stable — at most nPhase passes)
//   3. Write back.
//
// For the common case of small nPhase (≤ 16) the local arrays fit in
// registers.  nPhase is a runtime value; max supported = 32.
// ---------------------------------------------------------------------------
__global__ void k_gibbs_simplex_N(
    Real** d_fields,
    int nPhase,
    int nx, int ny, int nz,
    int sx, int sy,
    int g)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));

    // Read into local array (max 32 phases)
    Real phi[32];
    for (int i = 0; i < nPhase; ++i)
        phi[i] = d_fields[i][idx];

    // Iterative clipping: each pass handles one (or more) negative phase(s)
    for (int pass = 0; pass < nPhase; ++pass) {
        // Accumulate total deficit from negative phases
        double deficit = 0.0;
        int    nPos    = 0;
        for (int i = 0; i < nPhase; ++i) {
            if (phi[i] < 0.0) { deficit -= phi[i]; phi[i] = 0.0; }
            else              { ++nPos; }
        }
        if (deficit == 0.0) break;   // nothing negative — done

        // Distribute deficit equally among positive phases
        if (nPos == 0) {
            // Edge case: all phases clipped to 0 → assign everything to first
            phi[0] = 1.0;
            break;
        }
        double share = deficit / nPos;
        for (int i = 0; i < nPhase; ++i) {
            if (phi[i] > 0.0) phi[i] -= share;
        }
        // A positive phase may have gone negative after subtraction;
        // loop continues to fix that.
    }

    // Write back
    for (int i = 0; i < nPhase; ++i)
        d_fields[i][idx] = phi[i];
}

// ---------------------------------------------------------------------------
// Internal launcher
// ---------------------------------------------------------------------------
static void launch(const std::vector<ScalarField*>& fields)
{
    if (fields.empty()) return;

    const int nPhase = static_cast<int>(fields.size());
    if (nPhase > 32)
        throw std::runtime_error("gibbsSimplexOnGPU: max 32 phases supported");

    // Validate that all fields share the same layout
    const ScalarField& ref = *fields[0];
    for (int i = 1; i < nPhase; ++i) {
        if (fields[i]->storedDims[0] != ref.storedDims[0] ||
            fields[i]->storedDims[1] != ref.storedDims[1] ||
            fields[i]->storedDims[2] != ref.storedDims[2] ||
            fields[i]->ghost         != ref.ghost)
            throw std::runtime_error(
                "gibbsSimplexOnGPU: all fields must share the same mesh layout");
        if (!fields[i]->deviceAllocated())
            throw std::runtime_error(
                "gibbsSimplexOnGPU: field '" + fields[i]->name
                + "' has no device memory allocated");
    }

    // Build a host-side array of device pointers, copy it to the device
    std::vector<Real*> h_ptrs(nPhase);
    for (int i = 0; i < nPhase; ++i)
        h_ptrs[i] = fields[i]->d_curr;

    Real** d_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, nPhase * sizeof(Real*)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs.data(),
                          nPhase * sizeof(Real*),
                          cudaMemcpyHostToDevice));

    // Grid covers interior cells (ghost cells are left untouched)
    const int nx = ref.mesh.n[0];
    const int ny = ref.mesh.n[1];
    const int nz = ref.mesh.n[2];
    const int sx = ref.storedDims[0];
    const int sy = ref.storedDims[1];
    const int g  = ref.ghost;

    const dim3 blk(8, 8, 4);
    const dim3 grd((nx + blk.x - 1) / blk.x,
                   (ny + blk.y - 1) / blk.y,
                   (nz + blk.z - 1) / blk.z);

    k_gibbs_simplex_N<<<grd, blk>>>(d_ptrs, nPhase, nx, ny, nz, sx, sy, g);
    PHIX_KERNEL_CHECK("gibbsSimplexOnGPU");

    CUDA_CHECK(cudaFree(d_ptrs));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void gibbsSimplexOnGPU(std::initializer_list<ScalarField*> fields)
{
    launch(std::vector<ScalarField*>(fields));
}

void gibbsSimplexOnGPU(const std::vector<ScalarField*>& fields)
{
    launch(fields);
}

} // namespace PhiX
