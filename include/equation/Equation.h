#pragma once

#include "equation/Term.h"
#include "field/ScalarField.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace PhiX {
// Forward declarations to avoid circular includes:
//   EvalPlan.h  includes  Equation.h  (via FieldOps.inl)
//   Equation.h  includes  EvalPlan.h  ← break this cycle by using fwd decls
struct ExprTree;
class  EvalPlan;
class  BoundaryCondition;
}

namespace PhiX {

// ---------------------------------------------------------------------------
// Equation
//
// Describes  d(unknown)/dt = RHS
// where RHS is an RHSExpr built from lap(), grad(), pw() terms.
//
// Workflow:
//   1. Construct with the unknown Field and any auxiliary fields / params.
//   2. Call setRHS(expr) once to define the equation.
//   3. Pass to a Solver, which calls computeRHS(rhs_field) each time step.
//
// computeRHS writes into the PHYSICAL cells of rhs (zero-initialises first),
// leaving ghost cells of rhs at zero — the Solver applies BCs before calling
// this function and is responsible for rhs memory lifecycle.
// ---------------------------------------------------------------------------

class Equation {
public:
    // Provide an explicit destructor so that the unique_ptr<EvalPlan>
    // can be compiled with an incomplete EvalPlan type in the header but
    // defined in Equation.cu where EvalPlan is fully visible.
    ~Equation();

    std::string              name;
    ScalarField&                   unknown;     // the d/dt field (non-owning ref)
    std::vector<ScalarField*>      auxFields;   // other fields used on RHS (non-owning)

    explicit Equation(ScalarField& unknown, const std::string& name = "");

    // -----------------------------------------------------------------------
    // Set the RHS expression.  Can be called again to update mid-simulation.
    // -----------------------------------------------------------------------
    void setRHS(const RHSExpr& expr);
    void setRHS(const Term& t);        // convenience: single-term RHS
    void setRHS(const ExprTree& tree); // ExprTree API (Stage 2+)

    // -----------------------------------------------------------------------
    // BC registration for ExprTree auto-injection (Stage 3).
    //
    // Associates a set of BCs with a ScalarField so that when setRHS(ExprTree)
    // encounters a composite ExprStencil node whose child references `field`,
    // the materialised scratch buffer gets the correct BCs applied
    // automatically (without the user calling lap(expr, bcs) explicitly).
    //
    // Example (a stencil applied to a composite child; expr_lap has no
    // ExprTree overload yet, so the ExprStencil node is built explicitly):
    //   eq.registerBC(mu, {&bc_periodic_x, &bc_periodic_y});
    //   ExprTree sum = ExprTree(mu) + ExprTree(c);
    //   ExprTree rhs(std::make_shared<ExprStencil>(
    //       StencilKind::LAP, /*axis=*/0, /*stencilWidth=*/1, sum.node));
    //   eq.setRHS(rhs);   // auto-injects mu's BCs on the scratch buffer
    // -----------------------------------------------------------------------
    void registerBC(const ScalarField& field,
                    std::vector<BoundaryCondition*> bcs);

    // -----------------------------------------------------------------------
    // Evaluate RHS into rhs.d_curr on GPU.
    // rhs must already have device memory allocated (rhs.allocDevice()).
    // -----------------------------------------------------------------------
    void computeRHS(ScalarField& rhs) const;

    // CPU fallback (useful for testing without a GPU)
    void computeRHSCPU(ScalarField& rhs) const;

    // -----------------------------------------------------------------------
    // Single-step advance methods
    //
    // Both methods apply `bcs` to `sourceField` (refreshing ghost cells)
    // before evaluating the RHS.  If `sourceField` is nullptr, BCs are
    // applied to `unknown` itself (backward-compatible default).
    //
    // advanceSteady   — STEADY assignment:  unknown  = RHS
    //                   Does NOT call advanceTimeLevelGPU; does NOT count time.
    //
    // advanceTransient — TRANSIENT Euler:   unknown += dt * RHS
    //                   Calls advanceTimeLevelGPU (d_prev ← d_curr) after update.
    //                   Increments step and time.
    // -----------------------------------------------------------------------
    void advanceSteady(const std::vector<BoundaryCondition*>& bcs,
                       ScalarField* sourceField = nullptr);

    void advanceTransient(const std::vector<BoundaryCondition*>& bcs,
                          double dt,
                          ScalarField* sourceField = nullptr);

    // Step counter and simulation time (incremented only by advanceTransient).
    int    step = 0;
    double time = 0.0;

    // [Stage 4] Attach a CUDA stream to this equation.  All GPU kernels and
    // memsets submitted by computeRHS / advanceTransient / advanceSteady will
    // use this stream.  nullptr (default) = CUDA default stream.
    // Caller is responsible for stream lifetime.
    void setStream(cudaStream_t s) { stream_ = s; }
    cudaStream_t stream() const   { return stream_; }

    // -----------------------------------------------------------------------
    // Convenience: return true if RHS has been set
    // -----------------------------------------------------------------------
    bool hasRHS() const { return !rhs_expr_.terms.empty() || (bool)eval_plan_; }
    int  requiredGhost() const { return requiredGhost_; }

private:
    RHSExpr               rhs_expr_;
    mutable ScratchPool   scratch_pool_;
    int                   requiredGhost_ = 0;

    // BC map for ExprTree auto-injection (Stage 3).
    std::unordered_map<const ScalarField*, std::vector<BoundaryCondition*>> bc_map_;

    // EvalPlan — set when equation is configured via setRHS(ExprTree).
    std::unique_ptr<EvalPlan> eval_plan_;

    // [Stage 4] CUDA stream for all GPU work submitted by this equation.
    cudaStream_t stream_ = nullptr;

    // Scratch field for advanceTransient (lazily allocated on first call).
    mutable std::unique_ptr<ScalarField> rhs_scratch_;
};

} // namespace PhiX
