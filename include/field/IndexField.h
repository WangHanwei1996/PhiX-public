#pragma once

#include "field/FieldLayout.h"
#include "mesh/Mesh.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace PhiX {

class ScalarField;

// ---------------------------------------------------------------------------
// IndexField
//
// Integer (int32) companion field for multi-grain / cellular-automaton
// semantics: grain IDs, phase indices, region labels.  Storage layout is
// identical to ScalarField (same ghost halo, same storedDims/planeOffset
// rules — including the 2D 3-plane z-ghost storage), so an IndexField can
// shadow a ScalarField cell-for-cell on both CPU and GPU.
//
// Values < 0 conventionally mean "unassigned" — the adopt/gather primitives
// below build on that convention.
//
// Single time level (labels have no prev), explicit lazy GPU buffer like
// ScalarField: allocDevice() then uploadToDevice() before use in kernels.
//
// Primitives (this header + IndexField.inl; nvcc-only like Term.h):
//
//   adoptFromNeighborGPU(idx, score, arrived[, out, table])
//       Unassigned cells whose `arrived(score[c])` predicate fires inherit
//       the index of their maximum-`score` assigned neighbour (full 8/26
//       neighbourhood, physical cells only).  The multi-grain solidification
//       front-adoption pattern (PoolSectionGPU k_gid).
//
//   gatherGPU(out, idx, table, fallback)
//       out[c] = idx[c] >= 0 ? table[idx[c]] : fallback  — expand a
//       per-index table (grain orientation, phase property) to a full
//       double field.
//
//   gatherPWGPU(out, idx, table, aux, fn)
//       out[c] = fn(idx[c], idx[c]>=0 ? table[idx[c]] : 0.0, aux[c]) —
//       gather with a user functor and one auxiliary field (orientation
//       colouring output etc.).  Use PHIX_FN for fn.
// ---------------------------------------------------------------------------

class IndexField {
public:
    std::string name;
    FieldLayout layout;

    const Mesh& mesh;
    int         ghost;
    int         storedDims[3];
    std::size_t storedSize;

    // CPU storage (single level)
    std::vector<int32_t> curr;

    // GPU storage (nullptr until allocDevice())
    int32_t* d_curr = nullptr;

    explicit IndexField(const Mesh& mesh,
                        const std::string& name,
                        int ghost = 1);

    // Non-copyable (owns GPU memory), movable
    IndexField(const IndexField&)            = delete;
    IndexField& operator=(const IndexField&) = delete;
    IndexField(IndexField&& other) noexcept;
    IndexField& operator=(IndexField&& other) noexcept;
    ~IndexField();

    // Same index mapping as ScalarField (physical OR ghost indices)
    inline int index(int i, int j, int k) const { return layout.index(i, j, k); }
    inline int index(int i, int j) const { return index(i, j, 0); }
    inline int index(int i)        const { return index(i, 0, 0); }

    inline std::size_t storedBytes() const { return storedSize * sizeof(int32_t); }
    inline std::size_t planeOffset() const { return layout.planeOffset(); }

    void fill(int32_t value);   // all stored cells (incl. ghost)

    bool deviceAllocated() const { return d_curr != nullptr; }
    void allocDevice();
    void freeDevice();
    void uploadToDevice()   const;
    void downloadFromDevice();
};

// ---------------------------------------------------------------------------
// DeviceTable — RAII device copy of a small per-index lookup table
// (grain-orientation table etc.).  Upload once, pass data() to the
// gather/adopt primitives every step.
// ---------------------------------------------------------------------------
class DeviceTable {
public:
    explicit DeviceTable(const std::vector<double>& host);

    DeviceTable(const DeviceTable&)            = delete;
    DeviceTable& operator=(const DeviceTable&) = delete;
    DeviceTable(DeviceTable&& other) noexcept;
    DeviceTable& operator=(DeviceTable&& other) noexcept;
    ~DeviceTable();

    const double* data() const { return d_; }
    std::size_t   size() const { return n_; }

private:
    double*     d_ = nullptr;
    std::size_t n_ = 0;
};

// ---------------------------------------------------------------------------
// gatherGPU — expand a per-index table into a double field:
//   out[c] = idx[c] >= 0 ? d_table[idx[c]] : fallback     (physical cells)
// idx and out must share mesh dims and ghost; d_table is a device pointer
// (DeviceTable::data()) large enough for every index present.
// ---------------------------------------------------------------------------
void gatherGPU(ScalarField& out, const IndexField& idx,
               const double* d_table, double fallback,
               cudaStream_t stream = nullptr);

} // namespace PhiX

// Template primitives (adoptFromNeighborGPU / gatherPWGPU) — kernel
// templates, instantiated by the caller.  Requires nvcc (like TermPW.inl).
#include "field/IndexField.inl"
