#pragma once

#include <string>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// IMaterial  —  abstract base for all material models
//
// Every concrete material must provide at minimum a human-readable name.
// Property-specific interfaces (thermodynamics, transport, …) are derived
// from this base so that heterogeneous material collections can be stored
// as  std::vector<std::unique_ptr<IMaterial>>.
// ---------------------------------------------------------------------------

class IMaterial {
public:
    virtual ~IMaterial() = default;

    /// Human-readable identifier, e.g. "Fe-B binary alloy".
    virtual std::string name() const = 0;
};

} // namespace Material
} // namespace PhiX
