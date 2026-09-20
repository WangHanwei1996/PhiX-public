#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"
#include "scheme/Isotropic.h"

namespace PhiX {

template<typename Scheme>
Term grad(const ScalarField& f, int axis, double coeff = 1.0);

// Default (CD2) overload
Term grad(const ScalarField& f, int axis, double coeff);

// Runtime-string scheme dispatch
// Supported names come from scheme::catalog: CD2 (default), CD4, CD6, Iso9
Term grad(const ScalarField& f, int axis, const std::string& scheme, double coeff = 1.0);

// Case-driven scheme (scheme/Schemes.h): grad(f, ax, sch, c) == grad(f, ax, sch.grad(f.name), c)
class Schemes;
Term grad(const ScalarField& f, int axis, const Schemes& sch, double coeff);   // default in Term.h

} // namespace PhiX
