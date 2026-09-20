#pragma once

// ---------------------------------------------------------------------------
// FusedTerm.h — Compile-time expression templates for fused CUDA kernels.
//
// Motivation:
//   The standard DSL (Term / RHSExpr) launches one CUDA kernel per term.
//   Composite terms such as  mul(mul(f, g), lap(h))  require materialising
//   intermediate results into global-memory scratch buffers — typically
//   20-30 kernel launches for a 10-term RHS like the MPF_AC_DW μ equation.
//
//   FusedTerm encodes the entire RHS expression in a C++ type, so a single
//   kernel template evaluates every term per-cell without touching global
//   memory for intermediates.
//
// Usage (in a .cu file compiled by nvcc):
//   #include "equation/FusedTerm.h"
//   using namespace PhiX::Fused;
//
//   auto rhs =
//       fpw2(phi0, phi_a, PHIX_FN(double a, double b){ return 2*W*a*b*b; }) +
//       fmul(ffield(phi0), fgrad_dot(phi_a, phi_a)) * (2.0*eps2) +
//       fmul(fmul(ffield(phi0), ffield(phi_a)), flap(phi_a)) * eps2 +
//       ...;
//
//   eq_mu0.setRHS(fuse(rhs, phi0));   // compiles to a single GPU kernel
//
// Notes:
//   • All node types are trivially copyable (safe to pass to CUDA kernels).
//   • The expression tree is embedded in the kernel parameter list.
//     Typical sizes: ~30 bytes/term × 10 terms ≈ 300 bytes — well within
//     the 4 KB CUDA kernel parameter limit.
//   • operator+ and operator*(double) are constrained via FusedNodeTag so
//     they do not interfere with PhiX's ScalarField arithmetic operators.
//   • Built-in nodes support CPU execution and bind current storage at launch.
//
// Requires: nvcc with --expt-extended-lambda (same as rest of PhiX).
// ---------------------------------------------------------------------------

#include "core/Check.h"
#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "equation/Term.h"   // Term, TermType, ScratchPool
#include "scheme/Schemes.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace PhiX {
namespace Fused {

// ============================================================================
// Tag — used for SFINAE on operator+ and operator*(double).
// Each node declares `using fused_node_tag = FusedNodeTag;` instead of
// inheriting, so aggregate initialization still works under C++14 semantics.
// ============================================================================

struct FusedNodeTag {};

namespace detail {
    template<typename T, typename = void>
    struct is_fused_node : std::false_type {};

    template<typename T>
    struct is_fused_node<T, std::void_t<typename T::fused_node_tag>> : std::true_type {};
} // namespace detail

// ============================================================================
// StencilParams — per-call mesh/stride context passed by value to eval()
// ============================================================================

struct StencilParams {
    int    sx, sy;              // storedDims[0], storedDims[1]
    Real   inv_dx2;             // 1 / dx²
    Real   inv_dy2;             // 1 / dy²  (0 if dim < 2)
    Real   inv_dz2;             // 1 / dz²  (0 if dim < 3)
    int    dim;
};

// ============================================================================
// Leaf nodes
// ============================================================================

// FieldNode: d_data[c]  (value at stored index c)
struct FieldNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_data;
    const ScalarField *source = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return d_data[c];
    }
};

// LapNode: ∇²f at stored index c  (2nd-order central FD)
struct LapNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_data;
    const ScalarField *source = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        Real v = (d_data[c + 1] - Real(2) * d_data[c] + d_data[c - 1]) * p.inv_dx2;
        if (p.dim >= 2)
            v += (d_data[c + p.sx] - Real(2) * d_data[c] + d_data[c - p.sx]) * p.inv_dy2;
        if (p.dim >= 3) {
            int ssz = p.sx * p.sy;
            v += (d_data[c + ssz] - Real(2) * d_data[c] + d_data[c - ssz]) * p.inv_dz2;
        }
        return v;
    }
};

// GradDotNode: ∇f · ∇g at stored index c  (central FD, all active axes)
struct GradDotNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f;
    const Real* d_g;
    const ScalarField *sourceF = nullptr;
    const ScalarField *sourceG = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        // (df/dx) · (dg/dx) = [(f[c+1]-f[c-1]) * (g[c+1]-g[c-1])] / (4 dx²)
        Real v = (d_f[c + 1] - d_f[c - 1]) * (d_g[c + 1] - d_g[c - 1])
                   * (Real(0.25) * p.inv_dx2);
        if (p.dim >= 2) {
            v += (d_f[c + p.sx] - d_f[c - p.sx]) *
                 (d_g[c + p.sx] - d_g[c - p.sx]) * (Real(0.25) * p.inv_dy2);
        }
        if (p.dim >= 3) {
            int ssz = p.sx * p.sy;
            v += (d_f[c + ssz] - d_f[c - ssz]) *
                 (d_g[c + ssz] - d_g[c - ssz]) * (Real(0.25) * p.inv_dz2);
        }
        return v;
    }
};

// ============================================================================
// Composite nodes  (templated — type encodes the expression tree structure)
// ============================================================================

// ScaleNode: coeff * inner
template<typename Inner>
struct ScaleNode {
    using fused_node_tag = FusedNodeTag;
    Inner  inner;
    Real   coeff;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return coeff * inner.eval(c, p);
    }
};

// MulNode: lhs * rhs  (element-wise product of two sub-expressions)
template<typename Lhs, typename Rhs>
struct MulNode {
    using fused_node_tag = FusedNodeTag;
    Lhs lhs;
    Rhs rhs;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return lhs.eval(c, p) * rhs.eval(c, p);
    }
};

// AddNode: lhs + rhs  (sum of two sub-expressions)
template<typename Lhs, typename Rhs>
struct AddNode {
    using fused_node_tag = FusedNodeTag;
    Lhs lhs;
    Rhs rhs;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return lhs.eval(c, p) + rhs.eval(c, p);
    }
};

// Pw1Node: fn(f[c])  — 1-field user functor
template<typename Fn>
struct Pw1Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f;
    Fn fn;
    const ScalarField *source = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f[c]);
    }
};

// Pw2Node: fn(f1[c], f2[c])  — 2-field user functor
template<typename Fn>
struct Pw2Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f1;
    const Real* d_f2;
    Fn fn;
    const ScalarField *source1 = nullptr;
    const ScalarField *source2 = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f1[c], d_f2[c]);
    }
};

// Pw3Node: fn(f1[c], f2[c], f3[c])  — 3-field user functor
template<typename Fn>
struct Pw3Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f1;
    const Real* d_f2;
    const Real* d_f3;
    Fn fn;
    const ScalarField *source1 = nullptr;
    const ScalarField *source2 = nullptr;
    const ScalarField *source3 = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f1[c], d_f2[c], d_f3[c]);
    }
};

// Pw4Node: fn(f1[c], f2[c], f3[c], f4[c])  — 4-field user functor
template<typename Fn>
struct Pw4Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f1;
    const Real* d_f2;
    const Real* d_f3;
    const Real* d_f4;
    Fn fn;
    const ScalarField *source1 = nullptr;
    const ScalarField *source2 = nullptr;
    const ScalarField *source3 = nullptr;
    const ScalarField *source4 = nullptr;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f1[c], d_f2[c], d_f3[c], d_f4[c]);
    }
};

// ============================================================================
// Operator overloads — constrained to FusedNodeTag types via SFINAE
// so they do not interfere with PhiX::ScalarField arithmetic operators.
// ============================================================================

// expr + expr  →  AddNode
template<typename L, typename R,
         typename = std::enable_if_t<
             detail::is_fused_node<L>::value ||
             detail::is_fused_node<R>::value>>
inline AddNode<L, R> operator+(L l, R r) { return {l, r}; }

// expr * coeff  →  ScaleNode
template <typename T, typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator*(T t, double c) {
    return {t, Real(c)};
}

// coeff * expr  →  ScaleNode
template <typename T, typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator*(double c, T t) {
    return {t, Real(c)};
}

// -expr  →  ScaleNode with coeff=-1
template<typename T,
         typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator-(T t) { return {t, -1.0}; }

// ============================================================================
// Factory functions
//
// Factories retain stable field identities; fields must outlive the expression.
// ============================================================================

// Field identity is retained on the host. Device/host storage is bound at launch.
inline FieldNode ffield(const ScalarField &f) {
    return {f.d_curr, &f};
}
inline LapNode flap(const ScalarField& f) {
    check::checkCartesian(f.mesh, "Fused::flap");
    check::checkGhost(f, 1, "Fused::flap");
    return {f.d_curr, &f};
}
inline GradDotNode fgrad_dot(const ScalarField& f, const ScalarField& g) {
    check::checkCartesian(f.mesh, "Fused::fgrad_dot");
    check::checkSameMesh(f, g, "Fused::fgrad_dot");
    check::checkGhost(f, 1, "Fused::fgrad_dot");
    check::checkGhost(g, 1, "Fused::fgrad_dot");
    return {f.d_curr, g.d_curr, &f, &g};
}

// Case-aware fixed implementations. Existing overloads remain explicit CD2
// choices. These overloads refuse a case requesting another discretisation;
// they do not silently replace a fused node with a different kernel.
inline LapNode flap(const ScalarField& f, const Schemes& schemes) {
    schemes.requireFixed("laplacian", "lap(" + f.name + ")", "CD2", "Fused::flap");
    return flap(f);
}

inline GradDotNode fgrad_dot(const ScalarField& f, const ScalarField& g,
                             const Schemes& schemes) {
    for (int axis = 0; axis < f.mesh.dim; ++axis)
        for (auto *field : {&f, &g}) {
            auto selected = schemes.gradient(field->name, axis);
            if (selected.name != "CD2")
                throw ValidationError("Fused::fgrad_dot", selected.describe(),
                                      "the explicit fused adapter requires CD2");
        }
    return fgrad_dot(f, g);
}

template<typename L, typename R>
inline MulNode<L, R> fmul(L l, R r) { return {l, r}; }

template<typename Fn>
inline Pw1Node<Fn> fpw(const ScalarField& f, Fn fn) {
    return {f.d_curr, fn, &f};
}

template<typename Fn>
inline Pw2Node<Fn> fpw2(const ScalarField& f1, const ScalarField& f2, Fn fn) {
    return {f1.d_curr, f2.d_curr, fn, &f1, &f2};
}

template<typename Fn>
inline Pw3Node<Fn> fpw3(const ScalarField& f1, const ScalarField& f2,
                        const ScalarField& f3, Fn fn) {
    return {f1.d_curr, f2.d_curr, f3.d_curr, fn, &f1, &f2, &f3};
}

template<typename Fn>
inline Pw4Node<Fn> fpw4(const ScalarField& f1, const ScalarField& f2,
                        const ScalarField& f3, const ScalarField& f4, Fn fn) {
    return {f1.d_curr, f2.d_curr, f3.d_curr, f4.d_curr, fn, &f1, &f2, &f3, &f4};
}

namespace detail {
inline const Real *current(const ScalarField *f, bool gpu, const Real *legacy) {
    if (!f)
        return legacy;
    if (gpu && !f->d_curr)
        throw std::runtime_error("Fused: input '" + f->name + "' is not on device");
    return gpu ? f->d_curr : f->curr.data();
}
template <class T, class = void> struct Binding {
    static T bind(T n, bool) { return n; }
    static void describe(const T &, OperationInfo &info) { info.complete = false; }
};
// Extensions can expose the same bind/describe contract without modifying this header.
template <class T> struct Binding<T, std::void_t<decltype(std::declval<T>().bound(true))>> {
    static T bind(T n, bool gpu) { return n.bound(gpu); }
    static void describe(const T &n, OperationInfo &info) { n.describe(info); }
};
template <> struct Binding<FieldNode> {
    static FieldNode bind(FieldNode n, bool gpu) {
        n.d_data = current(n.source, gpu, n.d_data);
        return n;
    }
    static void describe(const FieldNode &n, OperationInfo &info) {
        if (!n.source)
            info.complete = false;
        info.read(n.source, 0);
    }
};
template <> struct Binding<LapNode> {
    static LapNode bind(LapNode n, bool gpu) {
        n.d_data = current(n.source, gpu, n.d_data);
        return n;
    }
    static void describe(const LapNode &n, OperationInfo &info) {
        if (!n.source)
            info.complete = false;
        info.read(n.source, 1);
    }
};
template <> struct Binding<GradDotNode> {
    static GradDotNode bind(GradDotNode n, bool gpu) {
        n.d_f = current(n.sourceF, gpu, n.d_f);
        n.d_g = current(n.sourceG, gpu, n.d_g);
        return n;
    }
    static void describe(const GradDotNode &n, OperationInfo &info) {
        if (!n.sourceF)
            info.complete = false;
        info.read(n.sourceF, 1);
        if (!n.sourceG)
            info.complete = false;
        info.read(n.sourceG, 1);
    }
};
template <class Fn> struct Binding<Pw1Node<Fn>> {
    static Pw1Node<Fn> bind(Pw1Node<Fn> n, bool gpu) {
        n.d_f = current(n.source, gpu, n.d_f);
        return n;
    }
    static void describe(const Pw1Node<Fn> &n, OperationInfo &info) {
        if (!n.source)
            info.complete = false;
        info.read(n.source, 0);
    }
};
template <class Fn> struct Binding<Pw2Node<Fn>> {
    static Pw2Node<Fn> bind(Pw2Node<Fn> n, bool gpu) {
        n.d_f1 = current(n.source1, gpu, n.d_f1);
        n.d_f2 = current(n.source2, gpu, n.d_f2);
        return n;
    }
    static void describe(const Pw2Node<Fn> &n, OperationInfo &info) {
        if (!n.source1)
            info.complete = false;
        info.read(n.source1, 0);
        if (!n.source2)
            info.complete = false;
        info.read(n.source2, 0);
    }
};
template <class Fn> struct Binding<Pw3Node<Fn>> {
    static Pw3Node<Fn> bind(Pw3Node<Fn> n, bool gpu) {
        n.d_f1 = current(n.source1, gpu, n.d_f1);
        n.d_f2 = current(n.source2, gpu, n.d_f2);
        n.d_f3 = current(n.source3, gpu, n.d_f3);
        return n;
    }
    static void describe(const Pw3Node<Fn> &n, OperationInfo &info) {
        if (!n.source1)
            info.complete = false;
        info.read(n.source1, 0);
        if (!n.source2)
            info.complete = false;
        info.read(n.source2, 0);
        if (!n.source3)
            info.complete = false;
        info.read(n.source3, 0);
    }
};
template <class Fn> struct Binding<Pw4Node<Fn>> {
    static Pw4Node<Fn> bind(Pw4Node<Fn> n, bool gpu) {
        n.d_f1 = current(n.source1, gpu, n.d_f1);
        n.d_f2 = current(n.source2, gpu, n.d_f2);
        n.d_f3 = current(n.source3, gpu, n.d_f3);
        n.d_f4 = current(n.source4, gpu, n.d_f4);
        return n;
    }
    static void describe(const Pw4Node<Fn> &n, OperationInfo &info) {
        if (!n.source1)
            info.complete = false;
        info.read(n.source1, 0);
        if (!n.source2)
            info.complete = false;
        info.read(n.source2, 0);
        if (!n.source3)
            info.complete = false;
        info.read(n.source3, 0);
        if (!n.source4)
            info.complete = false;
        info.read(n.source4, 0);
    }
};
template <class L, class R> struct Binding<AddNode<L, R>> {
    static AddNode<L, R> bind(AddNode<L, R> n, bool gpu) {
        return {Binding<L>::bind(n.lhs, gpu), Binding<R>::bind(n.rhs, gpu)};
    }
    static void describe(const AddNode<L, R> &n, OperationInfo &info) {
        Binding<L>::describe(n.lhs, info);
        Binding<R>::describe(n.rhs, info);
    }
};
template <class L, class R> struct Binding<MulNode<L, R>> {
    static MulNode<L, R> bind(MulNode<L, R> n, bool gpu) {
        return {Binding<L>::bind(n.lhs, gpu), Binding<R>::bind(n.rhs, gpu)};
    }
    static void describe(const MulNode<L, R> &n, OperationInfo &info) {
        Binding<L>::describe(n.lhs, info);
        Binding<R>::describe(n.rhs, info);
    }
};
template <class T> struct Binding<ScaleNode<T>> {
    static ScaleNode<T> bind(ScaleNode<T> n, bool gpu) {
        return {Binding<T>::bind(n.inner, gpu), n.coeff};
    }
    static void describe(const ScaleNode<T> &n, OperationInfo &info) {
        Binding<T>::describe(n.inner, info);
    }
};
template <class T> OperationInfo describe(const T &n, const ScalarField &layout) {
    OperationInfo info;
    info.complete = true;
    info.execution = "fused";
    Binding<T>::describe(n, info);
    for (const auto &r : info.reads)
        if (r.cell) {
            check::checkSameMesh(*r.cell, layout, "Fused layout");
            check::checkGhost(*r.cell, r.halo, "Fused halo");
        }
    return info;
}
} // namespace detail

// ============================================================================
// Fused CUDA kernel — one thread per physical cell, evaluates the entire
// expression tree per-cell without intermediate global-memory writes.
// ============================================================================

template<typename Expr>
__global__ void kernel_fused_accumulate(
        Real* __restrict__ d_rhs,
        Expr expr,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx2, Real inv_dy2, Real inv_dz2,
        int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    StencilParams p{sx, sy, inv_dx2, inv_dy2, inv_dz2, dim};
    d_rhs[c] += expr.eval(c, p);
}

// ============================================================================
// 3-output fused kernel — one pass over the grid, writes three output fields.
//
// All three expressions share the same StencilParams and one launch. Each
// expression retains its own evaluation; cache reuse and compiler elimination
// of repeated loads are possible optimizations, not a guaranteed CSE contract.
// ============================================================================

template<typename E0, typename E1, typename E2>
__global__ void kernel_fused_multi3(
        Real* __restrict__ d_out0, E0 e0,
        Real* __restrict__ d_out1, E1 e1,
        Real* __restrict__ d_out2, E2 e2,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx2, Real inv_dy2, Real inv_dz2,
        int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    StencilParams p{sx, sy, inv_dx2, inv_dy2, inv_dz2, dim};

    // Evaluate all three expressions per cell before writing — compiler keeps
    // shared sub-expressions (lap(phi0), grad_dot, ...) in registers.
    d_out0[c] = e0.eval(c, p);
    d_out1[c] = e1.eval(c, p);
    d_out2[c] = e2.eval(c, p);
}

// fuse_multi_compute — launch one kernel to compute three output fields.
//
// `layout`  : any field on the same mesh (provides geometry / strides).
// `outN`    : output ScalarField; d_curr is written directly (no zero-init
//             needed — the full expression replaces the field value).
// `exprN`   : a Fused expression tree built with ffield/flap/fgrad_dot/fmul/
//             fpw2/fpw3/fpw4/operator+/operator*.  Build with `auto expr = ...;`
//
// Input fields referenced inside exprN must have ghost cells filled (BCs
// applied) before calling.  Typical usage in a time loop:
//
//   fuse_multi_compute(phi0,
//       mu0,  expr_mu0,
//       mu_a, expr_mu_a,
//       mu_b, expr_mu_b);
//
template<typename E0, typename E1, typename E2>
inline void fuse_multi_compute(
        const ScalarField& layout,
        ScalarField& out0, E0 expr0,
        ScalarField& out1, E1 expr1,
        ScalarField& out2, E2 expr2,
        cudaStream_t stream = nullptr)
{
    check::checkOnDevice(out0, "fuse_multi_compute");
    check::checkOnDevice(out1, "fuse_multi_compute");
    check::checkOnDevice(out2, "fuse_multi_compute");
    check::checkSameMesh(out0, layout, "fuse_multi_compute");
    check::checkSameMesh(out1, layout, "fuse_multi_compute");
    check::checkSameMesh(out2, layout, "fuse_multi_compute");

    OperationInfo info = detail::describe(expr0, layout);
    info.merge(detail::describe(expr1, layout));
    info.merge(detail::describe(expr2, layout));
    if (out0.d_curr == out1.d_curr || out0.d_curr == out2.d_curr || out1.d_curr == out2.d_curr)
        throw std::invalid_argument("fuse_multi_compute: duplicate output storage");
    for (const auto &r : info.reads)
        if (r.cell && (r.cell->d_curr == out0.d_curr || r.cell->d_curr == out1.d_curr ||
                       r.cell->d_curr == out2.d_curr))
            throw std::invalid_argument(
                "fuse_multi_compute: an output aliases an input; use separate output fields");
    const auto bound0 = detail::Binding<E0>::bind(expr0, true);
    const auto bound1 = detail::Binding<E1>::bind(expr1, true);
    const auto bound2 = detail::Binding<E2>::bind(expr2, true);

    int nx  = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int sx  = layout.storedDims[0], sy = layout.storedDims[1];
    int g   = layout.ghost;
    int dim = layout.mesh.dim;

    double inv_dx2 = 1.0 / (layout.mesh.d[0] * layout.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (layout.mesh.d[1] * layout.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (layout.mesh.d[2] * layout.mesh.d[2]) : 0.0;

    int blocks = (nx * ny * nz + 255) / 256;

    kernel_fused_multi3<E0, E1, E2>
        <<<blocks, 256, 0, stream>>>(out0.d_curr, bound0, out1.d_curr, bound1, out2.d_curr, bound2,
                                     nx, ny, nz, sx, sy, g, inv_dx2, inv_dy2, inv_dz2, dim);

    PHIX_KERNEL_CHECK("fuse_multi_compute");
}

// ============================================================================
// FusedRHSExpr<Expr>
//
// Wraps a fused expression and converts it to a single runtime Term so it
// integrates seamlessly with Equation::setRHS(const Term&).
//
// The Term's gpu_launcher captures the expression by value (safe because all
// Fused nodes are trivially copyable).
// ============================================================================

template<typename Expr>
class FusedRHSExpr {
public:
    FusedRHSExpr(Expr expr, const ScalarField& layout)
        : expr_(expr), layout_(&layout) {}

    // Implicit conversion to Term — lets Equation::setRHS accept FusedRHSExpr
    // directly without any changes to Equation.h / Equation.cu.
    operator Term() const {
        const ScalarField* lay = layout_;
        Expr  expr = expr_;

        int    nx  = lay->mesh.n[0];
        int    ny  = lay->mesh.n[1];
        int    nz  = lay->mesh.n[2];
        int    sx  = lay->storedDims[0];
        int    sy  = lay->storedDims[1];
        int    g   = lay->ghost;
        int    dim = lay->mesh.dim;
        double inv_dx2 = 1.0 / (lay->mesh.d[0] * lay->mesh.d[0]);
        double inv_dy2 = (dim >= 2) ? 1.0 / (lay->mesh.d[1] * lay->mesh.d[1]) : 0.0;
        double inv_dz2 = (dim >= 3) ? 1.0 / (lay->mesh.d[2] * lay->mesh.d[2]) : 0.0;
        int    total   = nx * ny * nz;

        Term t;
        t.type  = TermType::COMPOSITE;
        t.field = lay;
        t.coeff = 1.0;
        t.info = detail::describe(expr, *lay);
        for (const auto &r : t.info.reads)
            if (r.cell) {
                t.inputs.push_back(r.cell);
                t.ghostRequired = std::max(t.ghostRequired, r.halo);
            }
        t.rhsGhost = g;   // kernel indexes the rhs with the layout's strides

        // Wrap in a ScaleNode so the Term::coeff passed by computeRHS is
        // honoured correctly even if the user wraps the fused term.
        t.gpu_launcher = [expr, nx, ny, nz, sx, sy, g, inv_dx2, inv_dy2, inv_dz2, dim,
                          total](Real *d_rhs, double coeff, ScratchPool &pool) {
            auto scaled = ScaleNode<Expr>{detail::Binding<Expr>::bind(expr, true), Real(coeff)};
            int blocks  = (total + 255) / 256;
            kernel_fused_accumulate<ScaleNode<Expr>><<<blocks, 256, 0, pool.stream>>>(
                d_rhs, scaled, nx, ny, nz, sx, sy, g,
                inv_dx2, inv_dy2, inv_dz2, dim);
            PHIX_KERNEL_CHECK("fused");
        };

        const bool complete = t.info.complete;
        t.cpu_kernel = [=](Real *rhs, double coeff, ScratchPool &) {
            if (!complete)
                throw std::runtime_error("Fused CPU: opaque raw-pointer node has no host binding");
            const auto host = detail::Binding<Expr>::bind(expr, false);
            const StencilParams p{sx, sy, Real(inv_dx2), Real(inv_dy2), Real(inv_dz2), dim};
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const int c = (i + g) + sx * ((j + g) + sy * (k + g));
                        rhs[c] += Real(coeff) * host.eval(c, p);
                    }
        };

        return t;
    }

private:
    Expr               expr_;
    const ScalarField* layout_;
};

// Factory: fuse(expr, layout_field) → FusedRHSExpr<Expr>
// `layout_field` provides mesh/stride information; any field in the equation
// that lives on the same mesh works.
template<typename Expr>
inline FusedRHSExpr<Expr> fuse(Expr expr, const ScalarField& layout) {
    return FusedRHSExpr<Expr>(expr, layout);
}

} // namespace Fused
} // namespace PhiX
