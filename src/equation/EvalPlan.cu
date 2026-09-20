// ---------------------------------------------------------------------------
// EvalPlan.cu — Lowering pass: ExprTree → EvalPlan.
//
// See include/equation/EvalPlan.h for the design overview.
// ---------------------------------------------------------------------------

#include "equation/EvalPlan.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "equation/TermPW.inl"   // pw<Functor>() template definitions
#include "equation/FieldOps.inl" // detail::termTimesTerm / termTimesField
#include "boundary/BoundaryCondition.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PhiX {

// ===========================================================================
// EvalPlan::execute / executeCPU
// ===========================================================================

void EvalPlan::execute(ScalarField& rhs, ScratchPool& pool) const {
    for (const auto& s : steps) {
        if (!s.gpu_launcher)
            throw std::runtime_error("EvalPlan::execute: a step has no GPU launcher");
        s.gpu_launcher(rhs.d_curr, s.coeff, pool);
    }
}

void EvalPlan::executeCPU(ScalarField& rhs, ScratchPool& pool) const {
    for (const auto& s : steps) {
        if (!s.cpu_kernel)
            throw std::runtime_error("EvalPlan::executeCPU: a step has no CPU kernel");
        s.cpu_kernel(rhs.curr.data(), s.coeff, pool);
    }
}

// ===========================================================================
// Internal lowering helpers
// ===========================================================================

// Convert an EvalStep.kind from a Term's type.
static EvalStep::Kind kindFromTerm(const Term& t) {
    return (t.type == TermType::LAPLACIAN || t.type == TermType::GRADIENT)
               ? EvalStep::Kind::STENCIL
               : EvalStep::Kind::LOCAL;
}

// Wrap a Term into an EvalStep.
static EvalStep stepFromTerm(Term t) {
    EvalStep s;
    s.source = t;
    s.kind         = kindFromTerm(t);
    s.coeff        = t.coeff;
    s.field        = t.field;
    s.gpu_launcher = std::move(t.gpu_launcher);
    s.cpu_kernel   = std::move(t.cpu_kernel);
    return s;
}

// Forward declarations of the two recursive helpers.
static void    lowerToSteps  (const ExprNode*, double coeff,
                               const ScalarField* layout,
                               const BcMap& bc_map,
                               std::vector<EvalStep>& out);
static RHSExpr lowerToRHSExpr(const ExprNode*, double coeff,
                               const ScalarField* layout,
                               const BcMap& bc_map);

// Lookup BCs for a representative field; returns empty vector if not found.
static std::vector<BoundaryCondition*>
lookupBcs(const ScalarField* rep, const BcMap& bc_map)
{
    if (!rep) return {};
    auto it = bc_map.find(rep);
    if (it == bc_map.end()) return {};
    return it->second;
}

// ---------------------------------------------------------------------------
// lowerToRHSExpr — lower a subtree to an RHSExpr (multiple Terms).
// Used when a subtree appears as an operand inside ExprMul.
// ---------------------------------------------------------------------------
static RHSExpr lowerToRHSExpr(const ExprNode* n, double coeff,
                               const ScalarField* layout,
                               const BcMap& bc_map)
{
    // ExprAdd flattens directly.
    if (auto* add = dynamic_cast<const ExprAdd*>(n)) {
        RHSExpr out;
        out += lowerToRHSExpr(add->left.get(),  coeff, layout, bc_map);
        out += lowerToRHSExpr(add->right.get(), coeff, layout, bc_map);
        return out;
    }

    // All other nodes: lower to EvalSteps, then pack into Terms.
    std::vector<EvalStep> sub_steps;
    lowerToSteps(n, coeff, layout, bc_map, sub_steps);

    RHSExpr out;
    for (auto& s : sub_steps) {
        Term t = s.source;
        out.terms.push_back(std::move(t));
    }
    return out;
}

// ---------------------------------------------------------------------------
// lowerToSteps — main recursive lowering function.
// Appends EvalSteps to `out`.
// `coeff` is the accumulated outer coefficient.
// `layout` is the representative ScalarField for mesh/ghost queries (may be
// nullptr; caller must guarantee it is set for nodes that need it).
// `bc_map` provides BCs for composite stencil expressions (Stage 3).
// ---------------------------------------------------------------------------
static void lowerToSteps(const ExprNode* n, double coeff,
                         const ScalarField* layout,
                         const BcMap& bc_map,
                         std::vector<EvalStep>& out)
{
    if (!n) return;

    // ── ExprScale ────────────────────────────────────────────────────────────
    if (auto* sc = dynamic_cast<const ExprScale*>(n)) {
        const ScalarField* child_lay = sc->child->repField();
        lowerToSteps(sc->child.get(), coeff * sc->coeff,
                     child_lay ? child_lay : layout, bc_map, out);
        return;
    }

    // ── ExprNeg ──────────────────────────────────────────────────────────────
    if (auto* neg = dynamic_cast<const ExprNeg*>(n)) {
        lowerToSteps(neg->child.get(), -coeff, layout, bc_map, out);
        return;
    }

    // ── ExprAdd ──────────────────────────────────────────────────────────────
    if (auto* add = dynamic_cast<const ExprAdd*>(n)) {
        lowerToSteps(add->left.get(),  coeff, layout, bc_map, out);
        lowerToSteps(add->right.get(), coeff, layout, bc_map, out);
        return;
    }

    // ── ExprLeaf ─────────────────────────────────────────────────────────────
    if (auto* leaf = dynamic_cast<const ExprLeaf*>(n)) {
        Term t = pw(*leaf->field,
                    [] __host__ __device__ (double v) { return v; },
                    coeff);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprScalar ───────────────────────────────────────────────────────────
    if (auto* cst = dynamic_cast<const ExprScalar*>(n)) {
        if (!layout)
            throw std::logic_error(
                "lowerExprTree: ExprScalar node with no layout field in context");
        double val = coeff * cst->value;
        Term t = pw(*layout,
                    [val] __host__ __device__ (double) { return val; },
                    1.0);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprMul ──────────────────────────────────────────────────────────────
    if (auto* mul_n = dynamic_cast<const ExprMul*>(n)) {
        const ScalarField* lay = mul_n->repField();
        if (!lay) lay = layout;
        if (!lay)
            throw std::logic_error("lowerExprTree: ExprMul has no layout field");

        RHSExpr left_expr  = lowerToRHSExpr(mul_n->left.get(),  1.0, lay, bc_map);
        RHSExpr right_expr = lowerToRHSExpr(mul_n->right.get(), 1.0, lay, bc_map);

        Term mul_term = detail::termTimesTerm(left_expr, right_expr, *lay, coeff);
        out.push_back(stepFromTerm(std::move(mul_term)));
        return;
    }

    // ── ExprStencil ──────────────────────────────────────────────────────────
    if (auto* st = dynamic_cast<const ExprStencil*>(n)) {
        // Peel Scale/Neg wrappers to get the effective child and coefficient.
        double leaf_coeff = coeff;
        const ExprNode* child = st->child.get();

        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(child)) {
                leaf_coeff *= sc->coeff;
                child = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(child)) {
                leaf_coeff = -leaf_coeff;
                child = neg->child.get();
            } else {
                break;
            }
        }

        if (auto* leaf = dynamic_cast<const ExprLeaf*>(child)) {
            // Simple leaf case — no BC materialisation needed.
            Term t;
            switch (st->kind) {
            case StencilKind::LAP:
                t = st->schemeName.empty() ? lap(*leaf->field, leaf_coeff)
                                           : lap(*leaf->field, st->schemeName, leaf_coeff);
                break;
            case StencilKind::GRAD:
                t = st->schemeName.empty()
                        ? grad(*leaf->field, st->axis, leaf_coeff)
                        : grad(*leaf->field, st->axis, st->schemeName, leaf_coeff);
                break;
            case StencilKind::ISO_GRAD:
                t = iso_grad(*leaf->field, st->axis, leaf_coeff);
                break;
            default:
                throw std::logic_error(
                    "lowerExprTree: unsupported StencilKind in ExprStencil");
            }
            if (!st->selection.empty())
                t.info.schemes.push_back(st->selection);
            out.push_back(stepFromTerm(std::move(t)));
            return;
        }

        // ── Composite child: requires BC injection (Stage 3) ────────────────
        // Lower the full (un-peeled) child to an RHSExpr.
        // leaf_coeff already accounts for Scale/Neg peeling, and the
        // child pointer still points to the inner node after peeling.
        // But we need to lower the original st->child (including the peeled
        // Scale/Neg) and let leaf_coeff absorb only the wrapper coefficients.
        //
        // Simpler: lower the original child with coeff=1.0 and fold
        // leaf_coeff into the outer stencil coeff.
        const ScalarField* rep = st->child->repField();
        auto bcs = lookupBcs(rep, bc_map);

        if (bcs.empty() && !bc_map.empty()) {
            // No BCs registered for this field — warn via logic_error.
            throw std::logic_error(
                "lowerExprTree: ExprStencil on composite expression requires BCs "
                "but none were found in the BcMap for the representative field. "
                "Call Equation::registerBC() before setRHS().");
        }
        if (bcs.empty()) {
            // No bc_map provided at all — use Stage 2 error.
            throw std::logic_error(
                "lowerExprTree: ExprStencil applied to a composite child "
                "expression requires BC auto-injection. "
                "Pass a BcMap to lowerExprTree, or use "
                "lap(expr, bcs) / grad(expr, axis, bcs) from the Term API.");
        }

        // Lower the original child sub-tree to an RHSExpr (coeff=1 here;
        // leaf_coeff is applied through the stencil call).
        RHSExpr child_expr = lowerToRHSExpr(child, 1.0, rep, bc_map);

        Term t;
        switch (st->kind) {
        case StencilKind::LAP:
            t = st->schemeName.empty() ? lap(child_expr, bcs, leaf_coeff)
                                       : lap(child_expr, bcs, st->schemeName, leaf_coeff);
            break;
        case StencilKind::GRAD:
            t = st->schemeName.empty()
                    ? grad(child_expr, st->axis, bcs, leaf_coeff)
                    : grad(child_expr, st->axis, bcs, st->schemeName, leaf_coeff);
            break;
        case StencilKind::ISO_GRAD:
            t = iso_grad(child_expr, st->axis, bcs, leaf_coeff);
            break;
        default:
            throw std::logic_error(
                "lowerExprTree: unsupported StencilKind for composite child");
        }
        if (!st->selection.empty())
            t.info.schemes.push_back(st->selection);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprStencilBinary ────────────────────────────────────────────────────
    if (auto* sb = dynamic_cast<const ExprStencilBinary*>(n)) {
        if (sb->kind != StencilKind::GRAD_DOT)
            throw std::logic_error(
                "lowerExprTree: unsupported ExprStencilBinary kind");

        double leaf_coeff = coeff;
        const ExprNode* l_node = sb->left.get();
        const ExprNode* r_node = sb->right.get();

        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(l_node)) {
                leaf_coeff *= sc->coeff;
                l_node = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(l_node)) {
                leaf_coeff = -leaf_coeff;
                l_node = neg->child.get();
            } else { break; }
        }
        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(r_node)) {
                leaf_coeff *= sc->coeff;
                r_node = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(r_node)) {
                leaf_coeff = -leaf_coeff;
                r_node = neg->child.get();
            } else { break; }
        }

        auto* lf = dynamic_cast<const ExprLeaf*>(l_node);
        auto* rf = dynamic_cast<const ExprLeaf*>(r_node);

        if (!lf || !rf)
            throw std::logic_error(
                "lowerExprTree: GRAD_DOT requires both children to be plain "
                "ScalarField leaves (Stage 3+ will lift this restriction)");

        Term t = grad_dot(*lf->field, *rf->field, leaf_coeff);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprPointwise1 ───────────────────────────────────────────────────────
    if (auto *pw = dynamic_cast<const ExprPointwise1 *>(n)) {
        if (!pw->termCapture)
            throw std::logic_error("ExprPointwise: missing term capture");
        out.push_back(stepFromTerm(pw->termCapture->term * coeff));
        return;
    }

    throw std::logic_error(
        "lowerExprTree: unknown or unhandled ExprNode type encountered");
}

// ===========================================================================
// lowerExprTree — public entry points
// ===========================================================================
EvalPlan lowerExprTree(const ExprTree& tree) {
    static const BcMap empty_map;
    return lowerExprTree(tree, empty_map);
}

EvalPlan lowerExprTree(const ExprTree& tree, const BcMap& bc_map) {
    validateGhostRequirements(tree);

    if (!tree.node)
        throw std::logic_error("lowerExprTree: empty ExprTree");

    const ScalarField* layout = tree.repField();

    EvalPlan plan;
    lowerToSteps(tree.node.get(), 1.0, layout, bc_map, plan.steps);
    return plan;
}

} // namespace PhiX
