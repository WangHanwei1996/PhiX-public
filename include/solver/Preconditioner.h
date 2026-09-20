#pragma once

// ---------------------------------------------------------------------------
// Preconditioner.h — preconditioner concept for the matrix-free CG layer.
//
// A preconditioner approximates z ≈ A⁻¹ r for the layer's system
//
//     A = alpha·I − sigma·L
//
// and must be SPD for PCG.  alpha and sigma arrive as DEVICE pointers (the
// ConjugateGradient scalar slots): every kernel a preconditioner launches
// reads them through the pointer, so a captured CG burst graph stays valid
// across per-solve coefficient changes — the same contract the operator
// kernels follow (LinearSolver.h).
//
// Provided preconditioners:
//   DiagonalPreconditioner  : z = r / (alpha − sigma·diagL) with diagL the
//                             cell diagonal of L — constant scalar (a no-op
//                             rescaling for constant-coefficient operators,
//                             interface smoke) or a per-cell field (the
//                             variable-coefficient case it exists for).
//   ChebyshevPreconditioner : fixed-degree Chebyshev approximation of A⁻¹ on
//                             [alpha, alpha + sigma·lamMaxL], lamMaxL a bound
//                             on |λ(L)|.  Degree-k apply = k operator
//                             applications, no inner products — graph-
//                             friendly, cuts PCG iterations ~k× on stiff σ.
//
// Both are streamSafe when their ingredients are, so they run inside the CG
// burst graph.  See test/moduleTest/solver/test_precond.cu.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "solver/LinearSolver.h"   // LinearOperator

#include <memory>

namespace PhiX {

// ===========================================================================
// Preconditioner — z ≈ (alpha·I − sigma·L)⁻¹ r, matrix-free.
// apply() may mutate r's GHOST cells only (BC refresh by an inner operator);
// physical r is read-only.  z is overwritten.
// ===========================================================================
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void apply(ScalarField& r, ScalarField& z,
                       const double* d_alpha, const double* d_sigma,
                       cudaStream_t stream = nullptr) = 0;

    // True when every launch honours the stream argument — precondition for
    // running inside the CG burst graph.
    virtual bool streamSafe() const { return false; }

    // Halo width needed on r/z.
    virtual int ghostRequired() const { return 0; }
};

// ===========================================================================
// DiagonalPreconditioner — z = r / (alpha − sigma·diagL).
// ===========================================================================
class DiagonalPreconditioner : public Preconditioner {
public:
    // Constant cell diagonal of L (e.g. LaplacianOp: −2D·Σ 1/dxᵢ² − shift).
    explicit DiagonalPreconditioner(double diagL);

    // Per-cell diagonal of L from a device-resident field (non-owning; the
    // caller keeps it alive and refreshed).  Must be ≤ 0 where A is SPD.
    explicit DiagonalPreconditioner(const ScalarField* diagL);

    void apply(ScalarField& r, ScalarField& z,
               const double* d_alpha, const double* d_sigma,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override { return true; }

private:
    double             diagConst_ = 0.0;
    const ScalarField* diagField_ = nullptr;
};

// ===========================================================================
// ChebyshevPreconditioner — degree-K Chebyshev approximation of A⁻¹.
//
// lamMaxL bounds the spectral radius of L (|λ(L)| ≤ lamMaxL; L negative
// semidefinite), so spec(A) ⊂ [alpha, alpha + sigma·lamMaxL].  The interval
// and the three-term recurrence scalars are recomputed ON DEVICE from the
// alpha/sigma slots at every apply — per-solve sigma changes propagate
// through a captured graph.  For LaplacianOp, lamMaxL = 4·D·Σ 1/dxᵢ² + shift
// (CD2 bound); helper below.
// ===========================================================================
class ChebyshevPreconditioner : public Preconditioner {
public:
    ChebyshevPreconditioner(LinearOperator& L, double lamMaxL, int degree,
                            const Mesh& mesh, int ghost);
    ~ChebyshevPreconditioner();

    ChebyshevPreconditioner(const ChebyshevPreconditioner&)            = delete;
    ChebyshevPreconditioner& operator=(const ChebyshevPreconditioner&) = delete;

    void apply(ScalarField& r, ScalarField& z,
               const double* d_alpha, const double* d_sigma,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override { return L_.streamSafe(); }
    int  ghostRequired() const override { return L_.ghostRequired(); }

    // CD2 spectral bound of a LaplacianOp on this mesh: 4·D·Σ 1/dxᵢ² + shift.
    static double lambdaMaxCD2(const Mesh& mesh, double D, double shift = 0.0);

private:
    LinearOperator& L_;
    double          lamMaxL_;
    int             degree_;
    ScalarField     t_, d_;      // L·z scratch, increment
    double*         d_c_ = nullptr;   // device recurrence slots
};

} // namespace PhiX
