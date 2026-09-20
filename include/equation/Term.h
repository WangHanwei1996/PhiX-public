#pragma once
#include "equation/Operation.h"

// ---------------------------------------------------------------------------
// Term.h — Expression DSL for composing RHS of time-evolution equations.
//
// Requires nvcc compilation (CUDA device code is referenced by templates).
//
// Scalar DSL usage:
//   eq.setRHS(M * lap(phi) + M * pw(phi, PHIX_FN (double p) {
//       return p - p*p*p; }));
//
// Vector DSL usage:
//   veq.setRHS(nu * lap(v));            // lap(VectorField) -> VectorRHSExpr
//   veq.setRHS(grad(p));                // grad(ScalarField) -> VectorRHSExpr
//   RHSExpr divV = div(v);              // div(VectorField) -> RHSExpr
//   VectorRHSExpr curlV = curl(v);      // curl(VectorField) -> VectorRHSExpr (3D)
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

// Convenience macro for pw() lambdas — avoids repeating __host__ __device__
// Usage:  pw(c, PHIX_FN (double c_val) { return ...; })
#define PHIX_FN [=] __host__ __device__

#include <cstddef>
#include <functional>
#include <vector>

namespace PhiX {

class BoundaryCondition;

// ---------------------------------------------------------------------------
// ScratchPool
//
// Recyclable buffer pool used by composite Term launchers (Term*Term,
// lap(expr,bcs), etc.) to materialise intermediate results without repeated
// cudaMalloc/cudaFree.  Each Equation owns one pool; Equation::computeRHS
// resets it before evaluating the RHS.
//
// All sizes are in `Real` elements (not bytes).  Buffers are NOT zeroed by
// acquire — the caller is responsible for cudaMemset / std::fill.
// ---------------------------------------------------------------------------
class ScratchPool {
public:
    ScratchPool() = default;
    ~ScratchPool();

    ScratchPool(const ScratchPool&)            = delete;
    ScratchPool& operator=(const ScratchPool&) = delete;

    Real* acquireDevice(std::size_t size);
    Real* acquireHost  (std::size_t size);

    void reset() { next_dev_ = 0; next_host_ = 0; }

    // [Stage 4] Stream to use for all GPU operations issued through this pool.
    // nullptr = CUDA default stream (backwards-compatible, in-order execution).
    // Set by Equation::computeRHS before executing steps.
    cudaStream_t stream = nullptr;

private:
    std::vector<Real*>               dev_bufs_;
    std::vector<std::size_t>         dev_sizes_;
    std::size_t                      next_dev_  = 0;

    std::vector<std::vector<Real>>   host_bufs_;
    std::size_t                      next_host_ = 0;
};

// ---------------------------------------------------------------------------
// TermLauncher — host-side std::function that launches (or runs) one term.
//
//   args: (rhs_data, effective_coeff, scratch_pool)
//
// The launcher captures all source field pointers at construction time
// (capturing `const ScalarField*` so that pointer-swap tricks like the one
// used by RK4 still work — d_curr is re-read on every invocation).
//
// `pool` provides recyclable scratch buffers for nested expressions; it is
// reset once per Equation::computeRHS call.  Simple Terms (lap, grad, pw)
// ignore it.
// ---------------------------------------------------------------------------
using TermLauncher = std::function<void(Real* /*rhs*/,
                                        double /*coeff*/,
                                        ScratchPool& /*pool*/)>;

enum class TermType { LAPLACIAN, GRADIENT, POINTWISE, COMPOSITE };

// ---------------------------------------------------------------------------
// Term — one additive contribution to an RHS expression
// ---------------------------------------------------------------------------
struct Term {
    TermType      type  = TermType::LAPLACIAN;
    double        coeff = 1.0;
    // Reference field used by computeRHS only for null/device-allocation
    // sanity checks.  Composite terms (Term*Term, lap(expr,bcs), ...) point
    // to a representative source field that is guaranteed to be on device.
    const ScalarField*  field = nullptr;
    // Every ScalarField this term READS (v3.7.0).  computeRHS zeroes its
    // target before accumulating, so a term whose input aliases the target
    // would read zeros -- a silent-wrong-answer trap.  computeRHS uses this
    // list (plus `field`) to detect the alias and route the evaluation
    // through a scratch buffer instead.  Factories reading more than
    // `field` must list all their ScalarField inputs here.
    std::vector<const ScalarField*> inputs;
    OperationInfo info;
    int           axis  = 0;         // for GRADIENT: 0=x, 1=y, 2=z
    int           ghostRequired = 0; // stencil width required by this term
    // Ghost width this term ASSUMES for the rhs buffer it writes into, or -1
    // if it uses the source field's layout (the default).  divFace indexes
    // the rhs with the face ghost; computeRHS validates the match.
    int           rhsGhost = -1;

    TermLauncher  gpu_launcher;   // host fn that launches GPU kernel
    TermLauncher  cpu_kernel;     // pure CPU fallback

    // Coefficient arithmetic — scale coeff, reuse launchers unchanged
    Term  operator*(double s) const { Term t = *this; t.coeff *= s; return t; }
    Term  operator/(double s) const { return *this * (1.0 / s); }
    Term  operator-()         const { return *this * (-1.0); }
};

// Declare a complete contract only at factories that know every input.
inline void describeInputs(Term &t, int halo = 0, bool corners = false) {
    t.info.complete = true;
    for (auto *f : t.inputs)
        t.info.read(f, halo, corners);
}
inline OperationInfo operationInfo(const Term &t) {
    auto info = t.info;
    // Legacy lists remain useful for alias checks, but do not certify purity.
    if (!info.complete)
        info.read(t.field, t.ghostRequired);
    for (auto *f : t.inputs)
        info.read(f);
    return info;
}
inline void inheritInputs(Term &out, const Term &child) {
    out.info.merge(operationInfo(child));
    for (const auto &r : out.info.reads)
        if (r.cell && std::find(out.inputs.begin(), out.inputs.end(), r.cell) == out.inputs.end())
            out.inputs.push_back(r.cell);
    out.ghostRequired = std::max(out.ghostRequired, child.ghostRequired);
}

inline Term operator*(double s, const Term& t) { return t * s; }
inline Term operator/(const Term& t, double s) { return t * (1.0 / s); }

// ---------------------------------------------------------------------------
// RHSExpr — ordered sum of Terms  (rhs = sum_i term_i)
// ---------------------------------------------------------------------------
struct RHSExpr {
    std::vector<Term> terms;

    RHSExpr() = default;
    explicit RHSExpr(const Term& t) { terms.push_back(t); }

    RHSExpr& operator+=(const Term& t)    { terms.push_back(t);  return *this; }
    RHSExpr& operator-=(const Term& t)    { terms.push_back(-t); return *this; }
    RHSExpr& operator+=(const RHSExpr& e) {
        terms.insert(terms.end(), e.terms.begin(), e.terms.end());
        return *this;
    }
    RHSExpr& operator-=(const RHSExpr& e) {
        for (const auto& t : e.terms) terms.push_back(-t);
        return *this;
    }

    RHSExpr operator+(const Term& t)    const { RHSExpr r=*this; r+=t; return r; }
    RHSExpr operator-(const Term& t)    const { RHSExpr r=*this; r-=t; return r; }
    RHSExpr operator+(const RHSExpr& e) const { RHSExpr r=*this; r+=e; return r; }
    RHSExpr operator-(const RHSExpr& e) const { RHSExpr r=*this; r-=e; return r; }
    RHSExpr operator*(double s)         const {
        RHSExpr r;
        for (const auto& t : terms) r.terms.push_back(t * s);
        return r;
    }
};

inline OperationInfo operationInfo(const RHSExpr &e) {
    OperationInfo info;
    info.complete = true;
    for (const auto &t : e.terms)
        info.merge(operationInfo(t));
    return info;
}
inline void inheritInputs(Term &out, const RHSExpr &e) {
    for (const auto &t : e.terms)
        inheritInputs(out, t);
}

// Free overloads so users can write: a + b, a - b, s * expr
inline RHSExpr operator+(const Term& a, const Term& b) { RHSExpr e(a); e += b; return e; }
inline RHSExpr operator-(const Term& a, const Term& b) { RHSExpr e(a); e -= b; return e; }
inline RHSExpr operator+(const Term& t, const RHSExpr& e) { RHSExpr r(t); r += e; return r; }
inline RHSExpr operator-(const Term& t, const RHSExpr& e) { RHSExpr r(t); r -= e; return r; }
inline RHSExpr operator*(double s, const RHSExpr& e) { return e * s; }

// ---------------------------------------------------------------------------
// Built-in differential operator factories.
//
// Scalar lap/grad are implemented in operators/* and default to CD2 in this
// version.  Composite-expression overloads remain in Equation.cu because they
// depend on scratch-pool materialisation.
// ---------------------------------------------------------------------------

// coeff * nabla^2(f)   — 2nd-order central FD Laplacian summed over active axes
Term lap(const ScalarField& f, double coeff = 1.0);

// Case-driven scheme (scheme/Schemes.h): scheme looked up by field name.
// Declared here too because most solvers reach lap/grad through Term.h.
class Schemes;
Term lap(const ScalarField& f, const Schemes& sch, double coeff = 1.0);
Term grad(const ScalarField& f, int axis, const Schemes& sch, double coeff = 1.0);

// coeff * d(f)/d(x_axis)  — 2nd-order central FD component gradient
Term grad(const ScalarField& f, int axis, double coeff = 1.0);

// coeff * d(f)/d(x_axis) — 9-point isotropic gradient (Patra-Karttunen, 2D).
// When composed with another grad to form a Laplacian/divergence the
// resulting stencil is the 9-point isotropic Laplacian, which dramatically
// reduces grid anisotropy for problems like dendritic solidification.
// For 1D / 3D meshes (or axis == 2) falls back to grad() with a one-time
// [PhiX] WARNING on stderr.
Term iso_grad(const ScalarField& f, int axis, double coeff = 1.0);

// iso_grad on composite expressions (materialises, applies BCs, then 9-pt stencil).
// Identical to grad(Term, axis, bcs) but uses the isotropic stencil.
// Falls back to grad(expr, axis, bcs) for non-2D or axis >= 2.
Term iso_grad(const Term&    t, int axis,
              const std::vector<BoundaryCondition*>& bcs,
              double coeff = 1.0);
Term iso_grad(const RHSExpr& e, int axis,
              const std::vector<BoundaryCondition*>& bcs,
              double coeff = 1.0);

// ---------------------------------------------------------------------------
// grad_dot(f, g, coeff=1.0) — pointwise dot product of gradients: ∇f · ∇g
//
//   rhs[idx] += coeff * Σ_a  (df/dx_a)[idx] * (dg/dx_a)[idx]
//
// Uses 2nd-order central FD on all active axes.  Requires ghost cells on
// both fields (standard BC application before time-stepping is sufficient).
// Both fields must share the same mesh dimensions and ghost width.
//
// Typical use in GFA-type models:
//   grad_dot(phi_i, phi_j)          // |∇φᵢ · ∇φⱼ|
//   mul(phi_j, grad_dot(phi_i, phi_j), -eps_ij*eps_ij)
// ---------------------------------------------------------------------------
Term grad_dot(const ScalarField& f, const ScalarField& g, double coeff = 1.0);
Term grad_dot(const ScalarField &f, const ScalarField &g, const Schemes &, double coeff = 1.0);

// ---------------------------------------------------------------------------
// Differential operators on composite expressions (Term / RHSExpr).
//
// The expression is materialised into a scratch ScalarField at evaluation
// time, BCs are applied to its ghost cells, then the standard finite-
// difference stencil is applied.  Required because expressions do not own
// ghost cells and the stencil needs them.
//
// `bcs` typically matches the BCs used for the field that drives the
// expression (e.g. the same BCs you would apply to `mu` if the expression
// is a closed-form rewrite of `mu`).
//
// TODO(vector): add VectorRHSExpr overloads of lap/grad/div on expressions
// when vector solvers need them.
// ---------------------------------------------------------------------------
Term lap (const Term&    t,
          const std::vector<BoundaryCondition*>& bcs,
          double coeff = 1.0);
Term lap (const RHSExpr& expr,
          const std::vector<BoundaryCondition*>& bcs,
          double coeff = 1.0);

Term grad(const Term&    t,    int axis,
          const std::vector<BoundaryCondition*>& bcs,
          double coeff = 1.0);
Term grad(const RHSExpr& expr, int axis,
          const std::vector<BoundaryCondition*>& bcs,
          double coeff = 1.0);

// Explicit scheme selection for a materialized intermediate. BCs belong to that
// intermediate; the new System::define + bc interface is preferred for reuse.
Term lap(const Term &, const std::vector<BoundaryCondition *> &, const std::string &scheme,
         double coeff = 1);
Term lap(const RHSExpr &, const std::vector<BoundaryCondition *> &, const std::string &scheme,
         double coeff = 1);
Term grad(const Term &, int axis, const std::vector<BoundaryCondition *> &,
          const std::string &scheme, double coeff = 1);
Term grad(const RHSExpr &, int axis, const std::vector<BoundaryCondition *> &,
          const std::string &scheme, double coeff = 1);

// grad(expr) without axis -> VectorRHSExpr (one component per mesh axis).
struct VectorRHSExpr;
VectorRHSExpr grad(const Term&    t,
                   const std::vector<BoundaryCondition*>& bcs,
                   double coeff = 1.0);
VectorRHSExpr grad(const RHSExpr& expr,
                   const std::vector<BoundaryCondition*>& bcs,
                   double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — pointwise user-defined transform
//
// coeff * Functor()(phi) applied element-wise to the physical cells of f.
//
// Functor requirements:
//   double operator()(double phi) const     <- CPU path
//   __device__ double operator()(double) const  <- GPU path
//
// Example functor (double-well derivative for Allen-Cahn):
//   struct DW { __device__ double operator()(double p) const
//                { return p*(1.0-p*p); } };
//
// Or with extended lambdas (nvcc --expt-extended-lambda):
//   pw(phi, [] __device__ (double p) { return p*(1.0-p*p); })
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f, Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 2-field pointwise transform
//   rhs[idx] += coeff * func(f1[idx], f2[idx])
//   Functor: __host__ __device__ double operator()(double, double) const
//   Both fields must share the same mesh and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2,
        Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 3-field pointwise transform
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx])
//   Functor: __host__ __device__ double operator()(double, double, double) const
//   All fields must share the same mesh and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 4-field pointwise transform
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx], f4[idx])
//   Functor: __host__ __device__ double operator()(double, double,
//                                                  double, double) const
//   All fields must share the same mesh and ghost width.
//   (Anti-trapping-style terms need (phi, U, dphi, n̂) at once — P-3.)
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 5..8-field pointwise transforms (v3.7.0)
//
//   rhs[idx] += coeff * func(f1[idx], ..., fN[idx])
//
// Motivation: multi-physics right-hand sides routinely read more than four
// fields at once.  The bicrystal directional-solidification solver
// (Tourret-Karma model) needs (psi, U, thT, p, psi_x, psi_y) in ONE
// functor; under the 4-field cap it had to materialise ~15 intermediate
// fields whose only job was ferrying values between pw calls.
// All fields must share the same mesh and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, const ScalarField& f5,
        Functor func, double coeff = 1.0);

template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, const ScalarField& f5, const ScalarField& f6,
        Functor func, double coeff = 1.0);

template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, const ScalarField& f5, const ScalarField& f6,
        const ScalarField& f7, Functor func, double coeff = 1.0);

template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, const ScalarField& f5, const ScalarField& f6,
        const ScalarField& f7, const ScalarField& f8,
        Functor func, double coeff = 1.0);

// ===========================================================================
// VectorRHSExpr — per-component RHS expression for vector equations
//
// VectorRHSExpr wraps N RHSExpr objects, one per vector component.
// Supports the same coefficient arithmetic as RHSExpr:
//   double * VectorRHSExpr, VectorRHSExpr +/- VectorRHSExpr, etc.
//
// Typical use:
//   VectorEquation veq(v, "diffusion");
//   veq.setRHS(nu * lap(v));     // lap returns VectorRHSExpr
// ===========================================================================

struct VectorRHSExpr {
    std::vector<RHSExpr> components;

    VectorRHSExpr() = default;
    explicit VectorRHSExpr(int n) : components(n) {}

    int nComponents() const { return static_cast<int>(components.size()); }

    RHSExpr&       operator[](int c)       { return components[c]; }
    const RHSExpr& operator[](int c) const { return components[c]; }

    VectorRHSExpr& operator+=(const VectorRHSExpr& o) {
        for (int c = 0; c < nComponents(); ++c) components[c] += o.components[c];
        return *this;
    }
    VectorRHSExpr& operator-=(const VectorRHSExpr& o) {
        for (int c = 0; c < nComponents(); ++c) components[c] -= o.components[c];
        return *this;
    }
    VectorRHSExpr operator+(const VectorRHSExpr& o) const {
        VectorRHSExpr r = *this; r += o; return r;
    }
    VectorRHSExpr operator-(const VectorRHSExpr& o) const {
        VectorRHSExpr r = *this; r -= o; return r;
    }
    VectorRHSExpr operator*(double s) const {
        VectorRHSExpr r(nComponents());
        for (int c = 0; c < nComponents(); ++c) r[c] = components[c] * s;
        return r;
    }
    VectorRHSExpr operator-() const { return (*this) * (-1.0); }
};

inline VectorRHSExpr operator*(double s, const VectorRHSExpr& e) { return e * s; }

// ===========================================================================
// Vector operator factories
// ===========================================================================

// lap(VectorField)  — returns VectorRHSExpr;  component c is lap(vf[c])
VectorRHSExpr lap(const VectorField& vf, double coeff = 1.0);

// grad(ScalarField)  — returns VectorRHSExpr of dimension mesh.dim;
//   component c is grad(f, c)
VectorRHSExpr grad(const ScalarField& f, double coeff = 1.0);

// div(VectorField)  — returns RHSExpr (scalar);
//   result = sum_c grad(vf[c], c)
RHSExpr div(const VectorField& vf, double coeff = 1.0);

// div(VectorRHSExpr, bcs)  — divergence of an expression-valued vector field.
//   result = sum_c grad(v[c], c, bcs, coeff)
// `bcs` is applied to each materialised flux component before differentiation.
// TODO(vector): add a tensor-divergence overload returning VectorRHSExpr when
// vector solvers need it.
RHSExpr div(const VectorRHSExpr& v,
            const std::vector<BoundaryCondition*>& bcs,
            double coeff = 1.0);

// curl(VectorField)  — returns VectorRHSExpr (3D only, dim must be 3);
//   curl[0] = dv2/dy - dv1/dz
//   curl[1] = dv0/dz - dv2/dx
//   curl[2] = dv1/dx - dv0/dy
VectorRHSExpr curl(const VectorField& vf, double coeff = 1.0);

// pw(VectorField, Functor)  — pointwise per-component;  component c is pw(vf[c], func)
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, Functor func, double coeff = 1.0);

// pw(VectorField, ScalarField, Functor) — per-component binary op with scalar field
//   component c: rhs_c[idx] += coeff * func(vf[c][idx], sf[idx])
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, const ScalarField& sf,
                 Functor func, double coeff = 1.0);

// pw(VectorField, VectorField, Functor) — component-wise binary op
//   component c: rhs_c[idx] += coeff * func(vf1[c][idx], vf2[c][idx])
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf1, const VectorField& vf2,
                 Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw on Term expressions — materialise terms into scratch buffers, then apply
// a user functor pointwise over the physical cells.  Ghost cells are NOT
// required for pointwise operations, so no BCs are needed here.
// Mirrors the ScalarField pw overloads; accepts lazy Term expressions.
// ---------------------------------------------------------------------------

// pw(Term, Functor): rhs[idx] += coeff * func(materialise(t)[idx])
template<typename Functor>
Term pw(const Term& t, Functor func, double coeff = 1.0);

// pw(Term, Term, Functor): rhs[idx] += coeff * func(mat(t1)[idx], mat(t2)[idx])
template<typename Functor>
Term pw(const Term& t1, const Term& t2, Functor func, double coeff = 1.0);

// pw(Term, Term, Term, Functor)
template<typename Functor>
Term pw(const Term& t1, const Term& t2, const Term& t3, Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// Named field-multiplication functions — Hadamard (element-wise) product.
//
// Function form allows future multiplication variants (dot product, matrix-
// vector product, etc.) to coexist without operator overload ambiguity.
//
//   mul(a, b, coeff=1.0)  ≡  coeff * (a ⊙ b)   (pointwise)
//
// Accepts any combination of ScalarField, Term, and RHSExpr operands;
// mirrors all operator* overloads in FieldOps.inl.
// ---------------------------------------------------------------------------
Term mul(const ScalarField& f1, const ScalarField& f2, double coeff = 1.0);
Term mul(const Term&        t,  const ScalarField& f,  double coeff = 1.0);
Term mul(const ScalarField& f,  const Term&        t,  double coeff = 1.0);
Term mul(const Term&        t1, const Term&        t2, double coeff = 1.0);
Term mul(const RHSExpr&     e,  const ScalarField& f,  double coeff = 1.0);
Term mul(const ScalarField& f,  const RHSExpr&     e,  double coeff = 1.0);
Term mul(const RHSExpr&     e1, const RHSExpr&     e2, double coeff = 1.0);
Term mul(const Term&        t,  const RHSExpr&     e,  double coeff = 1.0);
Term mul(const RHSExpr&     e,  const Term&        t,  double coeff = 1.0);

// ---------------------------------------------------------------------------
// Dot product of two vector fields / expressions  →  RHSExpr (scalar)
//
//   dot(A, B, coeff=1.0)  ≡  coeff * Σ_c A[c] * B[c]   (pointwise sum)
//
// Four overloads for any combination of VectorField and VectorRHSExpr.
// ---------------------------------------------------------------------------
RHSExpr dot(const VectorField&    a, const VectorField&    b, double coeff = 1.0);
RHSExpr dot(const VectorRHSExpr&  a, const VectorField&    b, double coeff = 1.0);
RHSExpr dot(const VectorField&    a, const VectorRHSExpr&  b, double coeff = 1.0);
RHSExpr dot(const VectorRHSExpr&  a, const VectorRHSExpr&  b, double coeff = 1.0);

} // namespace PhiX

// Template definitions — included here so nvcc sees them in every TU that
// includes this header.  Contains __global__ code; requires nvcc.
#include "equation/TermPW.inl"
