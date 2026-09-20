#pragma once

#include "core/Error.h"
#include "equation/Equation.h"
#include "equation/Term.h"
#include "field/VectorField.h"
#include "boundary/BoundaryCondition.h"

#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// VectorEquation
//
// Describes  d(unknown)/dt = RHS  where unknown is a VectorField.
//
// Internally owns N scalar Equation objects (one per component).
// The RHS is expressed as a VectorRHSExpr where each component is an RHSExpr
// built from the usual lap / grad / pw / div / curl operators.
//
// Workflow:
//   1. Construct with the unknown VectorField.
//   2. Call setRHS(VectorRHSExpr) once to define the system.
//   3. Pass to a VectorSolver, which calls computeRHS(rhs_vf) each time step.
//
// computeRHS writes into the PHYSICAL cells of each component of rhs,
// zero-initialising first — same contract as scalar Equation::computeRHS.
// ---------------------------------------------------------------------------

class VectorEquation {
public:
    std::string    name;
    VectorField&   unknown;          // the d/dt field (non-owning ref)
    // NOTE: params["missing"] on the public map silently inserts 0.0 —
    // prefer the checked param() accessor below in solver code.
    std::map<std::string, double> params;  // named physical constants

    // Checked parameter lookup: throws ValidationError on a missing key,
    // listing the keys that do exist.
    double param(const std::string& key) const {
        auto it = params.find(key);
        if (it == params.end()) {
            std::string avail;
            for (const auto& kv : params)
                avail += (avail.empty() ? "" : ", ") + kv.first;
            throw ValidationError("VectorEquation::param",
                "no parameter named '" + key + "'",
                avail.empty() ? "the params map is empty"
                              : "available keys: " + avail);
        }
        return it->second;
    }

    explicit VectorEquation(VectorField& unknown,
                             const std::string& name = "");

    // -----------------------------------------------------------------------
    // Set the RHS expression.
    // expr.nComponents() must equal unknown.nComponents().
    // -----------------------------------------------------------------------
    void setRHS(const VectorRHSExpr& expr);

    // Per-component ExprTree path (Stage 2+ DSL).
    void setRHSComponent(int c, const ExprTree& tree);

    // -----------------------------------------------------------------------
    // Stream propagation (Stage 4)
    // Sets the CUDA stream used for all component equations' GPU launches.
    // Default stream_ = nullptr (CUDA default stream, in-order).
    // -----------------------------------------------------------------------
    void setStream(cudaStream_t s);
    cudaStream_t stream() const;

    // -----------------------------------------------------------------------
    // BC auto-injection (Stage 3)
    // Registers BCs for a scalar field in all component equations.
    // -----------------------------------------------------------------------
    void registerBC(const ScalarField& field,
                    std::vector<BoundaryCondition*> bcs);

    // -----------------------------------------------------------------------
    // Evaluate RHS into rhs on GPU.
    // -----------------------------------------------------------------------
    void computeRHS(VectorField& rhs) const;

    // CPU fallback
    void computeRHSCPU(VectorField& rhs) const;

    // -----------------------------------------------------------------------
    // One explicit forward-Euler time step (mirrors Equation::advanceTransient).
    // -----------------------------------------------------------------------
    void advanceTransient(const std::vector<BoundaryCondition*>& bcs, double dt);

    // -----------------------------------------------------------------------
    // Access individual component equations (for advanced use)
    // -----------------------------------------------------------------------
    Equation&       componentEquation(int c);
    const Equation& componentEquation(int c) const;

    bool hasRHS() const;

private:
    std::vector<std::unique_ptr<Equation>> equations_;  // one per component
};

} // namespace PhiX
