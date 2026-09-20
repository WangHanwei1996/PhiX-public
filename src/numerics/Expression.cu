#include "numerics/Expression.h"

namespace PhiX::numerics {
CellExpr<Fused::FieldNode> Spatial::value(const ScalarField &f) const {
    check2D(f);
    auto term = PhiX::pw(f, [] __host__ __device__(Real x) { return x; });
    return {Fused::ffield(f), std::move(term)};
}
} // namespace PhiX::numerics
