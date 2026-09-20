#pragma once

#include "boundary/BoundaryCondition.h"
#include "mesh/Mesh.h"

#include <nlohmann/json.hpp>
#include <memory>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// BCSet — owns a collection of BoundaryCondition objects and exposes raw
// pointers for use with Solver / SolverStep.
// ---------------------------------------------------------------------------
struct BCSet {
    std::vector<std::unique_ptr<BoundaryCondition>> storage;  ///< owns lifetime
    std::vector<BoundaryCondition*> ptrs;                     ///< non-owning view
};

// ---------------------------------------------------------------------------
// buildBCs — construct boundary conditions from a JSON config block.
//
// Expected JSON layout (2D example):
//   {
//       "x_min": "Periodic",                      // string shorthand
//       "x_max": "Periodic",
//       "y_min": "NoFlux",
//       "y_max": {"type": "Fixed", "value": 1.0}  // object form
//   }
//
// Each boundary entry may be either:
//   - a plain string:  "Periodic" | "NoFlux" | "Fixed" (value defaults to 0)
//   - an object:       {"type": "Fixed", "value": <double>}
//
// Rules:
//   - Periodic must be set on both sides of the same axis; mismatched
//     Periodic/non-Periodic on the same axis throws std::runtime_error.
//   - z_min / z_max are processed when present (3D problems).
//   - This helper resolves the legacy x_min/x_max style config onto the
//     Mesh default face patches (xmin/xmax/...); if a face has been split
//     into multiple sub-patches, callers must build BCs manually.
//
// Supported BC types: "Periodic", "NoFlux", "Fixed" (Dirichlet).
// ---------------------------------------------------------------------------
BCSet buildBCs(const Mesh& mesh, const nlohmann::json& bc_config);

} // namespace PhiX
