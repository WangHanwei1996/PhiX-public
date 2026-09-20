#pragma once
#include "numerics/Symbolic.h"
#include "scheme/Presets.h"

namespace PhiX::numerics::symbolic {
struct PreparationOptions {
    // Public data object: callers can compose their own bundle without adding
    // a manager or changing the solver. Output always lists resolved entries.
    scheme::Preset preset = scheme::preset("AxialCD2");
    nlohmann::json overrides = nlohmann::json::object();
};
class Preparation {
public:
    nlohmann::json schemes, report;
    std::string equations;
    // No mutation of the System. Files are an editable configuration and an
    // inspection report, not a serialized executable plan.
    void write(const std::string& directory) const;
};
Preparation prepare(const Expanded& expression,const ScalarField& layout,
                    const PreparationOptions& options = {});
// Shared with System::prepare; no evaluation or CUDA context is needed.
Preparation prepareInventory(const OperatorInventory&,const PreparationOptions&);
void inspectPreparation(Preparation&,const Expanded&,const ScalarField&);
} // namespace PhiX::numerics::symbolic
