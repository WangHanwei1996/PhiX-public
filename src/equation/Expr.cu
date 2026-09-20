// ---------------------------------------------------------------------------
// Expr.cu — Implementation of ExprTree operator factories.
//
// Stage 1: Each factory builds an Expr subtree AND a prebuilt Term (via the
// existing Term-based API) stored in a TermCapture.  This lets
// Equation::setRHS validate ghost requirements with the new tree while still
// delegating evaluation to the existing launcher path.
//
// Stage 2 will replace the TermCapture round-trip with proper EvalPlan
// lowering.
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "equation/Expr.h"
#include "equation/Term.h"
#include "scheme/Schemes.h"

#include <memory>

namespace PhiX {

// ===========================================================================
// expr_lap — ∇²f
// ===========================================================================
ExprTree expr_lap(const ScalarField& f, double coeff) {
    check::checkCartesian(f.mesh, "expr_lap");
    auto node = std::make_shared<ExprStencil>(
        StencilKind::LAP,
        /*axis=*/0,          // axis ignored for LAP
        /*stencilWidth=*/1,
        std::make_shared<ExprLeaf>(f)
    );
    ExprTree t(std::move(node));
    // Apply coefficient via ExprScale wrapper (constant folding handled there)
    if (coeff != 1.0)
        return t * coeff;
    return t;
}

// ===========================================================================
// expr_grad — d(f)/d(x_axis)
// ===========================================================================
ExprTree expr_grad(const ScalarField& f, int axis, double coeff) {
    check::checkCartesian(f.mesh, "expr_grad");
    auto node = std::make_shared<ExprStencil>(
        StencilKind::GRAD,
        axis,
        /*stencilWidth=*/1,
        std::make_shared<ExprLeaf>(f)
    );
    ExprTree t(std::move(node));
    if (coeff != 1.0)
        return t * coeff;
    return t;
}

// ===========================================================================
// expr_iso_grad — 9-point isotropic gradient component
// ===========================================================================
ExprTree expr_iso_grad(const ScalarField& f, int axis, double coeff) {
    check::checkCartesian(f.mesh, "expr_iso_grad");
    auto node = std::make_shared<ExprStencil>(
        StencilKind::ISO_GRAD,
        axis,
        /*stencilWidth=*/1,
        std::make_shared<ExprLeaf>(f)
    );
    ExprTree t(std::move(node));
    if (coeff != 1.0)
        return t * coeff;
    return t;
}

// ===========================================================================
// expr_grad_dot — ∇f · ∇g
// ===========================================================================
ExprTree expr_grad_dot(const ScalarField& f, const ScalarField& g, double coeff) {
    check::checkCartesian(f.mesh, "expr_grad_dot");
    check::checkSameMesh(f, g, "expr_grad_dot");
    auto node = std::make_shared<ExprStencilBinary>(
        StencilKind::GRAD_DOT,
        /*stencilWidth=*/1,
        std::make_shared<ExprLeaf>(f),
        std::make_shared<ExprLeaf>(g)
    );
    ExprTree t(std::move(node));
    if (coeff != 1.0)
        return t * coeff;
    return t;
}

ExprTree expr_lap(const ScalarField &f, const Schemes &sch, double coeff) {
    const auto selection = sch.select("laplacian", "lap(" + f.name + ")");
    auto node = std::make_shared<ExprStencil>(
        StencilKind::LAP, 0, scheme::find(scheme::Family::Laplacian, selection.name)->primaryHalo,
        std::make_shared<ExprLeaf>(f));
    node->schemeName = selection.name;
    node->selection = selection.describe();
    return ExprTree(node) * coeff;
}
ExprTree expr_grad(const ScalarField &f, int axis, const Schemes &sch, double coeff) {
    const auto selection = sch.select("gradient", "grad(" + f.name + ")");
    auto node = std::make_shared<ExprStencil>(
        StencilKind::GRAD, axis,
        scheme::find(scheme::Family::Gradient, selection.name)->primaryHalo,
        std::make_shared<ExprLeaf>(f));
    node->schemeName = selection.name;
    node->selection = selection.describe();
    return ExprTree(node) * coeff;
}

} // namespace PhiX
