#pragma once

// ---------------------------------------------------------------------------
// SemiImplicitSolver.h — first-order IMEX (implicit-explicit) time stepping.
//
// Evolves  dφ/dt = L φ + N(φ)  with the stiff linear generator L treated by
// backward Euler and the non-stiff/nonlinear remainder N by forward Euler:
//
//     (I − dt·L) φⁿ⁺¹ = φⁿ + dt·N(φⁿ)
//
// The implicit solve is the matrix-free CG layer (solver/LinearSolver.h);
// σ = dt stays outside the operator, so changing dt between steps costs
// nothing.  This removes the explicit stability limits that dominate
// phase-field runs:
//
//   Allen-Cahn / diffusion:  L = D∇²   (LaplacianOp)   — dt no longer ∝ dx²
//   Cahn-Hilliard:           L = −Mκ∇⁴ (BiharmonicOp)  — dt no longer ∝ dx⁴,
//       with N(c) = M∇²f'(c) (the classical linear-splitting scheme; for
//       very large dt add linear stabilisation to L as usual).
//
// The explicit part N is a plain PhiX Equation: set its RHS to ONLY the
// explicit terms (leave the stiff part out — it lives in L).  An Equation
// with no RHS set means N ≡ 0 (fully implicit linear step).
//
// Usage (Cahn-Hilliard):
//     Equation eqC(c, "c");
//     eqC.setRHS(lap(muE, M));           // explicit: M∇²f'(c), muE filled
//                                        // by the caller before advance()
//     BiharmonicOp Lc(M*kappa, {&bc}, {&bc});
//     SemiImplicitSolver semi(eqC, {&bc}, Lc, dt);
//     loop { fill muE = f'(c); bcMu.applyOnGPU(muE); semi.advance(); }
//
// Accuracy is first order in dt (backward/forward Euler pair).  BCs must be
// homogeneous (see LinearSolver.h) and are applied to the unknown before
// the explicit RHS evaluation and inside the operator during CG.
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "boundary/BoundaryCondition.h"
#include "boundary/BCBatch.h"
#include "field/ScalarField.h"
#include "solver/LinearSolver.h"

#include <functional>
#include <vector>

namespace PhiX {

// Options of the inner CG solve (top-level type: a nested struct's default
// member initialisers cannot appear in the enclosing class's default args).
struct SemiImplicitCGOptions {
    double relTol  = 1e-8;
    int    maxIter = 500;
};

class SemiImplicitSolver {
public:
    using CGOptions = SemiImplicitCGOptions;

    SemiImplicitSolver(Equation&                       eqExplicit,
                       std::vector<BoundaryCondition*> bcs,
                       LinearOperator&                 L,
                       double                          dt,
                       CGOptions                       cg = CGOptions());

    SemiImplicitSolver(const SemiImplicitSolver&)            = delete;
    SemiImplicitSolver& operator=(const SemiImplicitSolver&) = delete;

    double dt;             // may be changed between steps (operator reused)
    int    step = 0;
    double time = 0.0;

    void advance();

    void run(int nSteps,
             int callbackEvery = 0,
             std::function<void(const SemiImplicitSolver&)> callback = nullptr);

    // Diagnostics of the most recent implicit solve.
    const ConjugateGradient::Result& lastSolve() const { return last_; }

    const ScalarField& unknown() const { return eq_.unknown; }
    ScalarField&       unknown()       { return eq_.unknown; }

private:
    Equation&                       eq_;
    std::vector<BoundaryCondition*> bcs_;
    BCBatch                         bcBatch_;
    LinearOperator&                 L_;
    CGOptions                       cgOpts_;

    ScalarField        rhs_;   // N(φⁿ)
    ScalarField        b_;     // φⁿ + dt·N(φⁿ)
    ConjugateGradient  cg_;
    ConjugateGradient::Result last_;
};

} // namespace PhiX
