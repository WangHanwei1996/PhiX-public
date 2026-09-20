#pragma once

// ---------------------------------------------------------------------------
// LinearSolver.h — matrix-free linear-solver layer (GPU).
//
// The layer solves systems of the form
//
//     A x = b,     A = I − σ·L
//
// where L is a linear "stiff generator" represented matrix-free by a
// LinearOperator (no matrix is ever assembled — L is applied as stencil
// kernels).  This is exactly the system a backward-Euler / semi-implicit
// step needs (σ = dt), and keeping σ OUTSIDE the operator makes adaptive
// time steps free: the same operator serves every dt.
//
// Provided operators:
//   LaplacianOp  : L = D·∇²      (CD2; Allen-Cahn / diffusion stiff part)
//   BiharmonicOp : L = −G·∇²∇²   (two CD2 passes; Cahn-Hilliard ∇⁴ part,
//                                 A = I + σ·G·∇⁴)
//
// Both refresh the ghost cells of their input with the boundary conditions
// given at construction before applying the stencil.
//
// BC RESTRICTION (linearity): CG requires A to be linear, so only
// homogeneous BCs are admissible on the operator — PeriodicBC and NoFluxBC
// qualify; FixedBC only with value 0 (lift an inhomogeneous Dirichlet value
// into b yourself).  For pure periodic/no-flux problems A = I − σL is SPD
// and CG is the right Krylov method.
//
// Solver: ConjugateGradient — owns its scratch fields (construct once,
// solve every step).  Reductions (dot products) accumulate in double
// regardless of PHIX_PRECISION.
//
// Typical semi-implicit usage (see solver/SemiImplicitSolver.h for the
// packaged integrator):
//
//     LaplacianOp L(D, {&bcx, &bcy});
//     ConjugateGradient cg(mesh, ghost);
//     // per step:  b = phi + dt*N(phi);  solve (I - dt*L) phi_new = b
//     auto res = cg.solve(L, dt, phi, b, 1e-8, 500);
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "boundary/BoundaryCondition.h"
#include "boundary/BCBatch.h"

#include <memory>
#include <vector>

namespace PhiX {

// ===========================================================================
// LinearOperator — matrix-free y = L(x) over physical cells.
// apply() may mutate x's GHOST cells (BC refresh); physical x is untouched.
// ===========================================================================
class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    // `stream`: all launches (BC refresh + stencil) go to this stream
    // (nullptr = default).  Required for CUDA-graph capture.
    virtual void apply(ScalarField& x, ScalarField& y,
                       cudaStream_t stream = nullptr) = 0;

    // Halo width the operator's stencil needs on x and y.
    virtual int ghostRequired() const { return 1; }

    // True when every launch honours the stream argument (no fallback BCs
    // etc.) — precondition for graph capture.
    virtual bool streamSafe() const { return false; }
};

// ===========================================================================
// LaplacianOp — L = D·∇² − shift·I (CD2)
//
// The optional `shift` (>= 0) is the Eyre-style linear stabilisation for
// IMEX reaction terms: splitting  −W·g'(φ) = −W·s·φ + W·(s·φ − g'(φ))
// puts the destabilising linear part into L (shift = M·W·s) so the explicit
// remainder has a bounded slope.  A = I − dt·L stays SPD for shift >= 0.
// ===========================================================================
class LaplacianOp : public LinearOperator {
public:
    LaplacianOp(double D, std::vector<BoundaryCondition*> bcs,
                double shift = 0.0);

    void apply(ScalarField& x, ScalarField& y,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override {
        return !bcBatch_.built() || bcBatch_.fallbackCount() == 0;
    }

    double D() const { return D_; }
    double shift() const { return shift_; }

private:
    double                          D_;
    double                          shift_;
    std::vector<BoundaryCondition*> bcs_;
    BCBatch                         bcBatch_;   // built lazily on first apply
};

// ===========================================================================
// VarCoeffLaplacianOp — L = ∇·(a(x)∇·) with FACE coefficients, matrix-free
// (2D).  ax: (nx+1)·ny device doubles, ax[i + (nx+1)j] = a on the face
// between cells (i−1,j)|(i,j); ay: nx·(ny+1) analogously.  Non-owning views:
// the caller keeps the arrays alive and refreshed (a per-step coefficient
// update costs nothing here — pair with MultigridPreconditioner::setupFaces
// for the MG hierarchy).  Under periodic BCs supply ax[0] == ax[nx] (the
// wrap face, duplicated); under no-flux the ghost mirror zeroes the boundary
// gradient, so the boundary-face value is irrelevant (zero it for clarity —
// then the operator matches the MG level-0 stencil exactly).
// ===========================================================================
class VarCoeffLaplacianOp : public LinearOperator {
public:
    VarCoeffLaplacianOp(const Mesh& mesh, const double* d_ax,
                        const double* d_ay,
                        std::vector<BoundaryCondition*> bcs);

    void apply(ScalarField& x, ScalarField& y,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override {
        return !bcBatch_.built() || bcBatch_.fallbackCount() == 0;
    }

private:
    const double* ax_;
    const double* ay_;
    std::vector<BoundaryCondition*> bcs_;
    BCBatch                         bcBatch_;   // built lazily on first apply
};

// ===========================================================================
// BiharmonicOp — L = −G·∇²(∇²) so that A = I − σL = I + σ·G·∇⁴.
// bcsX   : BCs of the solution field (e.g. no-flux on c: ∂c/∂n = 0)
// bcsLap : BCs of the intermediate ∇²x (for Cahn-Hilliard the natural
//          choice is no-flux again: ∂(∇²c)/∂n = 0, i.e. zero μ-flux)
// The intermediate scratch field is allocated lazily on first apply.
// ===========================================================================
class BiharmonicOp : public LinearOperator {
public:
    BiharmonicOp(double G,
                 std::vector<BoundaryCondition*> bcsX,
                 std::vector<BoundaryCondition*> bcsLap);

    void apply(ScalarField& x, ScalarField& y,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override {
        return inner_.streamSafe() && outer_.streamSafe();
    }

    double G() const { return G_; }

private:
    double                          G_;
    std::vector<BoundaryCondition*> bcsX_, bcsLap_;
    LaplacianOp                     inner_, outer_;   // persistent (BC batches)
    std::unique_ptr<ScalarField>    lap_;   // scratch: ∇²x
};

class Preconditioner;   // solver/Preconditioner.h — z ≈ A⁻¹r for PCG

// ===========================================================================
// ConjugateGradient — solves (I − σ·L) x = b, matrix-free, on the GPU.
//
// Device-resident control flow (v2.20.0): α and β are computed by
// single-thread kernels from CUB reduction results that never leave the
// device — a CG iteration enqueues ~10 kernels and ZERO host round trips.
// The host only reads back the residual every `checkEvery` iterations to
// decide convergence (each readback is a full pipeline drain, which is
// what used to dominate the solve cost at small/medium grids).  The solve
// may therefore run up to checkEvery−1 iterations past the tolerance —
// a negligible price against the removed synchronisations.
//
// PCG (v3.5.2): pass a Preconditioner* M to run preconditioned CG — the
// classical rho = <r, z> recurrence with z = M⁻¹r; convergence stays on the
// TRUE residual ‖r‖/‖b‖ (one extra r·r reduction on the last iteration of
// each check burst).  M == nullptr keeps the plain-CG path bit-for-bit.
// The burst graph includes M when M->streamSafe().
// ===========================================================================
class ConjugateGradient {
public:
    struct Result {
        int    iterations  = 0;
        double relResidual = 0.0;   // ‖r‖₂ / ‖b‖₂ at exit
        bool   converged   = false;
    };

    // Convergence-check cadence (host readback every N iterations).
    int checkEvery = 4;

    // Capture one `checkEvery` burst as a CUDA graph and replay it
    // (collapses ~10·checkEvery kernel launches into one).  Auto-disabled
    // when the operator is not streamSafe(); the graph is re-captured
    // whenever x/b/operator change.  σ = dt lives in a device slot, so
    // per-step dt changes do NOT invalidate the graph.
    bool useGraph = true;

    // Scratch fields (r, p, Lp) are allocated once for this mesh/ghost.
    ConjugateGradient(const Mesh& mesh, int ghost);
    ~ConjugateGradient();

    ConjugateGradient(const ConjugateGradient&)            = delete;
    ConjugateGradient& operator=(const ConjugateGradient&) = delete;

    // x: initial guess in, solution out (its ghosts are mutated by L).
    // Throws std::runtime_error if maxIter is exhausted without reaching
    // relTol (set throwOnFail = false to get Result.converged == false
    // instead).
    Result solve(LinearOperator& L, double sigma,
                 ScalarField& x, const ScalarField& b,
                 double relTol = 1e-8, int maxIter = 500,
                 bool throwOnFail = true, Preconditioner* M = nullptr) {
        return solveOperator(L, 1.0, sigma, x, b, relTol, maxIter,
                             throwOnFail, M);
    }

    // Generalised system  A x = b,  A = alpha·I − sigma·L.
    // alpha = 1: the semi-implicit Helmholtz form above.
    // alpha = 0, sigma = 1, L = D∇²: pure Poisson −D∇²x = b (for pure
    // Neumann/periodic BCs use PoissonSolver, which handles the constant
    // nullspace).  A must be SPD on the solved subspace.
    Result solveOperator(LinearOperator& L, double alpha, double sigma,
                         ScalarField& x, const ScalarField& b,
                         double relTol = 1e-8, int maxIter = 500,
                         bool throwOnFail = true, Preconditioner* M = nullptr);

private:
    ScalarField r_, p_, Lp_;
    std::unique_ptr<ScalarField> z_;  // M⁻¹r scratch (allocated on first PCG)
    double*     d_s_ = nullptr;   // device scalars: rho, rhoNew, pAp, alphaCG,
                                  //                 betaCG, sigma, alphaOp, rr
    cudaStream_t stream_ = nullptr;   // all CG work runs here

    // Cached burst graph + the identity it was captured for
    cudaGraphExec_t graphExec_ = nullptr;
    const void* keyX_ = nullptr;
    const void* keyB_ = nullptr;
    const void* keyL_ = nullptr;
    const void* keyM_ = nullptr;
    int         keyBurst_ = 0;

    void enqueueIteration(LinearOperator& L, Preconditioner* M,
                          ScalarField& x, cudaStream_t stream, int n,
                          bool withRR);
    void destroyGraph_();
};

// ===========================================================================
// PoissonSolver — solves  −∇·(D∇Φ) = rhs  with homogeneous BCs, matrix-free
// (CG on A = −D∇²).  For pure Neumann/periodic BCs the operator has a
// constant nullspace: the RHS is projected to zero mean and the solution is
// returned mean-zero (set projectNullspace = false when the BCs fix the
// level, e.g. FixedBC(0)).
//
//     PoissonSolver poisson(mesh, ghost, epsPermittivity, {&bcx, &bcy});
//     poisson.solve(Phi, chargeDensity);        // BM6-style electrostatics
// ===========================================================================
class PoissonSolver {
public:
    PoissonSolver(const Mesh& mesh, int ghost, double D,
                  std::vector<BoundaryCondition*> bcs);

    PoissonSolver(const PoissonSolver&)            = delete;
    PoissonSolver& operator=(const PoissonSolver&) = delete;

    bool projectNullspace = true;

    ConjugateGradient::Result solve(ScalarField& phi, const ScalarField& rhs,
                                    double relTol = 1e-8, int maxIter = 1000,
                                    bool throwOnFail = true);

private:
    LaplacianOp        L_;
    ConjugateGradient  cg_;
    ScalarField        b_;      // mean-projected RHS copy
    double             cellCount_;
};

} // namespace PhiX
