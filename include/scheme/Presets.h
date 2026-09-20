#pragma once
#include "scheme/Schemes.h"

namespace PhiX::scheme {
// Named bundles are data over the normal operator catalogue. They neither
// own fields nor change continuous equations or select an approximate model.
struct Preset {
    std::string name, description;
    Schemes::Defaults defaults;
};
const std::vector<Preset>& presets();
const Preset& preset(const std::string& name);
} // namespace PhiX::scheme
