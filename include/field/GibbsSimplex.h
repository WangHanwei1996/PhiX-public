#pragma once

#include "field/ScalarField.h"

#include <initializer_list>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// gibbsSimplexOnGPU
//
// Apply Gibbs simplex projection to an arbitrary number of phase fields on
// the GPU.  For each interior cell the operation is:
//
//   1. Clip any negative phi_i to 0, distributing the deficit equally among
//      the remaining (positive) phases.
//   2. The sum phi_0 + phi_1 + ... is preserved if it was originally 1.
//
// All fields must share the same Mesh and ghost-cell count.
// All fields must have device memory allocated (allocDevice called).
//
// Overloads:
//   gibbsSimplexOnGPU({&phi0, &phi_a, &phi_b})   // initializer_list
//   gibbsSimplexOnGPU(fields_vec)                 // std::vector<ScalarField*>
// ---------------------------------------------------------------------------

void gibbsSimplexOnGPU(std::initializer_list<ScalarField*> fields);
void gibbsSimplexOnGPU(const std::vector<ScalarField*>& fields);

} // namespace PhiX
