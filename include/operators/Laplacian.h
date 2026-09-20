#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"
#include "scheme/Isotropic.h"

namespace PhiX {

template<typename Scheme>
Term lap(const ScalarField& f, double coeff = 1.0);

// Default (CD2) overload — no second default arg to avoid redefinition
Term lap(const ScalarField& f, double coeff);

// Convenience: runtime-string scheme dispatch
// Supported names come from scheme::catalog: CD2 (default), CD4, CD6, Iso9, Iso27
Term lap(const ScalarField& f, const std::string& scheme, double coeff = 1.0);

// Case-driven scheme (scheme/Schemes.h): lap(f, sch, c) == lap(f, sch.lap(f.name), c)
class Schemes;
Term lap(const ScalarField& f, const Schemes& sch, double coeff);   // default in Term.h

} // namespace PhiX
