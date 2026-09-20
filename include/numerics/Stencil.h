#pragma once
#include "numerics/Symbolic.h"
#include <vector>

namespace PhiX::numerics::symbolic {
// A stencil is data, not a field-owning kernel. Weights are dimensionless;
// FluxStencil applies physical mesh spacings exactly once.
struct StencilWeight { int x, y; double weight; };
using LinearStencil = std::vector<StencilWeight>;
Expr weightedSum(Expr value, const LinearStencil& stencil);
// For zero-sum derivative weights: subtract the first sampled value before
// accumulation. Constants vanish exactly. Translated paired stencils retain
// the same reference point and arithmetic at their common link midpoint.
Expr weightedDifferences(Expr value, const LinearStencil& stencil);

struct FluxLink {
    int x, y;
    double weight;
    LinearStencil coefficient, gradientX, gradientY;
};
// Difference matches Ji's published numerator; reconstructed projection uses
// the same vector in numerator and norm. Both are generic flux recipes.
enum class FluxProjection { LinkDifference, ReconstructedGradient };
struct FluxStencil {
    std::vector<FluxLink> links;
    bool squareRequired = false;
    FluxProjection projection = FluxProjection::LinkDifference;
    // Reusable for arbitrary scalar coefficients/potentials, including
    // pointwise expressions and derivatives. Opposite links must share the
    // same reconstruction to obtain a conservative custom recipe.
    Expr apply(Expr coefficient, Expr potential, double dx, double dy,
               bool normalize = false, double normFloor = 0) const;
    static FluxStencil axial();
    static FluxStencil isotropic(bool highOrderNormal = false);
};
} // namespace PhiX::numerics::symbolic
