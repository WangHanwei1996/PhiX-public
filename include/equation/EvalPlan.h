#pragma once

// ---------------------------------------------------------------------------
// EvalPlan.h — Lowered execution plan for ExprTree (Stage 2 of DSL refactor).
//
// An EvalPlan is produced by lowerExprTree() and executed by
// Equation::computeRHS() when the equation was set via setRHS(ExprTree).
//
// Stage 2 design (Strategy A — runtime fusion):
//   The lowering pass walks the ExprTree, converts each node to one or more
//   Term-based launchers (the existing GPU kernel infrastructure), and stores
//   them as an ordered list of EvalSteps.
//
//   Key structural split vs the raw RHSExpr approach:
//     • Steps are tagged as LOCAL or STENCIL so future passes can fuse/reorder.
//     • Ghost requirements are validated at lower-time, not at run-time.
//     • BC injection point is explicit (step holds a BC list, Stage 3 fills it).
//
//   Stage 2 execution is numerically identical to the old RHSExpr loop:
//   each step's launcher is called sequentially, accumulating into rhs.d_curr.
//   Actual kernel fusion is deferred to Stage 5 (Strategy B).
//
// Stage 3 (this release):
//   lowerExprTree gains a BcMap parameter.  When an ExprStencil wraps a
//   composite child (not a plain ExprLeaf), the lowering pass looks up the
//   BCs for the child's representative field in BcMap and invokes
//   lap/grad/iso_grad(RHSExpr, bcs, coeff) from the existing Term API.
//   Equation gains registerBC() and bcMap_ so setRHS(ExprTree) can pass the
//   map automatically.
//
// Future stages:
//   Stage 4: EvalPlan holds a cudaStream_t; execute() drops DeviceSynchronize.
//   Stage 5: LOCAL steps are fused into single templated kernels (Strategy B).
// ---------------------------------------------------------------------------

#include "equation/Term.h"
#include "equation/Expr.h"
#include "boundary/BoundaryCondition.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace PhiX {

// ===========================================================================
// BcMap — maps a source ScalarField to the BCs applied to it.
// Passed to lowerExprTree so that composite stencil expressions can have
// BCs applied to their materialised scratch buffer automatically.
// ===========================================================================
using BcMap = std::unordered_map<const ScalarField*,
                                  std::vector<BoundaryCondition*>>;

// ===========================================================================
// EvalStep — one unit of work in the plan
// ===========================================================================
struct EvalStep {
    enum class Kind {
        LOCAL,    // pointwise, no halo: leaf scale / pw / mul etc.
        STENCIL,  // needs ghost cells: lap, grad, grad_dot
    };

    Term source; // Preserve dependencies and layout when adapting back to Term.
    Kind   kind  = Kind::LOCAL;
    double coeff = 1.0;

    // GPU launcher and CPU fallback — same signature as Term::gpu_launcher.
    TermLauncher gpu_launcher;
    TermLauncher cpu_kernel;

    // Representative field (for layout queries and nullptr checks).
    const ScalarField* field = nullptr;

    // [Stage 3] BCs to apply to the scratch buffer before the stencil step.
    // Populated by lowerExprTree when a BcMap is provided.
    std::vector<BoundaryCondition*> bcs;
};

// ===========================================================================
// EvalPlan — ordered list of EvalSteps
// ===========================================================================
class EvalPlan {
public:
    EvalPlan() = default;

    std::vector<EvalStep> steps;

    bool empty() const { return steps.empty(); }
    RHSExpr expression() const {
        RHSExpr e;
        for (const auto &s : steps)
            e += s.source;
        return e;
    }

    // -----------------------------------------------------------------------
    // execute — accumulate all steps into rhs.d_curr on GPU.
    // rhs must already be zeroed and have device memory allocated.
    // -----------------------------------------------------------------------
    void execute(ScalarField& rhs, ScratchPool& pool) const;

    // -----------------------------------------------------------------------
    // executeCPU — CPU fallback (no device memory required).
    // rhs.curr must already be zeroed.
    // -----------------------------------------------------------------------
    void executeCPU(ScalarField& rhs, ScratchPool& pool) const;
};

// ===========================================================================
// lowerExprTree — convert an ExprTree into an EvalPlan.
//
// Two overloads:
//   (1) Without BcMap — composite stencil children throw std::logic_error.
//   (2) With BcMap   — composite stencil children resolve BCs automatically
//       from the map; missing entries cause std::logic_error only if the
//       child is genuinely composite (pure-leaf children never need BCs).
//
// Preconditions:
//   • validateGhostRequirements(tree) is called internally; callers need not
//     call it separately.
// ===========================================================================
EvalPlan lowerExprTree(const ExprTree& tree);
EvalPlan lowerExprTree(const ExprTree& tree, const BcMap& bc_map);

} // namespace PhiX

