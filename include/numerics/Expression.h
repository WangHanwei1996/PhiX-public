#pragma once
#include "equation/FusedTerm.h"
#include "equation/FieldOps.inl"
#include "equation/EvalPlan.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "operators/GradSq.h"
#include "operators/Advection.h"
#include "operators/Anisotropy.h"
#include "scheme/Isotropic.h"
#include <functional>

namespace PhiX::numerics {
enum class Fusion { Off, Auto, Required };
// Wrapping a functor declares that all field reads are arguments, with no side effects.
template <class Fn> struct Pure {
    Fn fn;
};
template <class Fn> Pure<Fn> pure(Fn fn) {
    return {fn};
}

inline Term sum(Term a, Term b) {
    if (a.field && b.field)
        check::checkSameMesh(*a.field, *b.field, "expression sum");
    Term t;
    t.type = TermType::COMPOSITE;
    t.field = a.field ? a.field : b.field;
    t.info.complete = true;
    inheritInputs(t, a);
    inheritInputs(t, b);
    t.rhsGhost = a.rhsGhost >= 0 ? a.rhsGhost : b.rhsGhost;
    if (a.rhsGhost >= 0 && b.rhsGhost >= 0 && a.rhsGhost != b.rhsGhost)
        throw std::invalid_argument("expression sum: incompatible output layouts");
    if (a.gpu_launcher && b.gpu_launcher)
        t.gpu_launcher = [=](Real *r, double c, ScratchPool &p) {
            a.gpu_launcher(r, c * a.coeff, p);
            b.gpu_launcher(r, c * b.coeff, p);
        };
    if (a.cpu_kernel && b.cpu_kernel)
        t.cpu_kernel = [=](Real *r, double c, ScratchPool &p) {
            a.cpu_kernel(r, c * a.coeff, p);
            b.cpu_kernel(r, c * b.coeff, p);
        };
    return t;
}
inline Term collapse(const RHSExpr &e) {
    if (e.terms.empty())
        throw std::invalid_argument("empty expression");
    Term t = e.terms.front();
    for (size_t i = 1; i < e.terms.size(); ++i)
        t = sum(t, e.terms[i]);
    return t;
}

// Sum independently lowered regions without losing each region's execution choice.
// Unlike CellExpr addition, this deliberately does not fuse across region boundaries.
template <class A, class B> struct RegionSum {
    A a;
    B b;
    Term lower(Fusion mode) const {
        auto x = a.lower(mode), y = b.lower(mode);
        auto t = sum(x, y);
        t.info.execution = "sum [" + x.info.execution + "; " + y.info.execution + "]";
        return t;
    }
};
template <class A, class B> auto sum(A a, B b) {
    return RegionSum<A, B>{std::move(a), std::move(b)};
}

namespace detail {
template <class Fn, std::size_t N> struct Pointwise {
    using fused_node_tag = Fused::FusedNodeTag;
    Fn fn;
    const ScalarField *fields[N];
    const Real *data[N] = {};
    Pointwise bound(bool gpu) const {
        auto n = *this;
        for (std::size_t i = 0; i < N; ++i)
            n.data[i] = Fused::detail::current(fields[i], gpu, data[i]);
        return n;
    }
    void describe(OperationInfo &info) const {
        for (const auto *f : fields) info.read(f);
    }
    template <std::size_t... I>
    __host__ __device__ Real apply(int c, std::index_sequence<I...>) const {
        return fn(data[I][c]...);
    }
    __host__ __device__ Real eval(int c, const Fused::StencilParams &) const {
        return apply(c, std::make_index_sequence<N>{});
    }
};
// A symbolic stencil has no device evaluator. Setup substitutes a compiled scheme.
template <bool Lap> struct Stencil {
    const ScalarField *field;
    int axis;
};
struct Opaque {};
template <class S, bool Lap> struct BoundStencil {
    using fused_node_tag = Fused::FusedNodeTag;
    const ScalarField *field;
    int axis;
    const Real *data = nullptr;
    Real invx, invy;
    BoundStencil bound(bool gpu) const {
        auto n = *this;
        n.data = Fused::detail::current(field, gpu, data);
        return n;
    }
    void describe(OperationInfo &info) const {
        info.read(field, S::ghostRequired(), std::string(S::name()) == "Iso9");
    }
    __host__ __device__ Real eval(int c, const Fused::StencilParams &p) const {
        if constexpr (Lap)
            return S::laplacian(data, c, p.sx, p.sy, 2, invx * invx, invy * invy, 0);
        else
            return S::gradient(data, c, axis, p.sx, p.sy, 2, invx, invy, 0);
    }
};
template <class N, class S> struct Compile {
    static auto get(N n) { return n; }
};
template <bool L, class S> struct Compile<Stencil<L>, S> {
    static auto get(Stencil<L> n) {
        return BoundStencil<S, L>{n.field, n.axis, nullptr, Real(1 / n.field->mesh.d[0]),
                                  Real(1 / n.field->mesh.d[1])};
    }
};
template <class N, class S> struct Compile<Fused::ScaleNode<N>, S> {
    static auto get(Fused::ScaleNode<N> n) {
        auto c = Compile<N, S>::get(n.inner);
        return Fused::ScaleNode<decltype(c)>{c, n.coeff};
    }
};
template <class L, class R, class S> struct Compile<Fused::AddNode<L, R>, S> {
    static auto get(Fused::AddNode<L, R> n) {
        auto l = Compile<L, S>::get(n.lhs);
        auto r = Compile<R, S>::get(n.rhs);
        return Fused::AddNode<decltype(l), decltype(r)>{l, r};
    }
};
template <class L, class R, class S> struct Compile<Fused::MulNode<L, R>, S> {
    static auto get(Fused::MulNode<L, R> n) {
        auto l = Compile<L, S>::get(n.lhs);
        auto r = Compile<R, S>::get(n.rhs);
        return Fused::MulNode<decltype(l), decltype(r)>{l, r};
    }
};
template <class N, class Fn> struct Transform {
    using fused_node_tag = Fused::FusedNodeTag;
    N child;
    Fn fn;
    Transform bound(bool gpu) const { return {Fused::detail::Binding<N>::bind(child, gpu), fn}; }
    void describe(OperationInfo &i) const { Fused::detail::Binding<N>::describe(child, i); }
    __host__ __device__ Real eval(int c, const Fused::StencilParams &p) const {
        return fn(child.eval(c, p));
    }
};
template <class N, class Fn, class S> struct Compile<Transform<N, Fn>, S> {
    static auto get(Transform<N, Fn> n) {
        auto c = Compile<N, S>::get(n.child);
        return Transform<decltype(c), Fn>{c, n.fn};
    }
};
} // namespace detail

// The typed fragment and its ordinary implementation describe the same formula.
// Type erasure happens only after the finite backend candidates have been compiled.
template <class Node, bool Fusible = true> struct CellExpr {
    Node node;
    Term reference;
    int stencils = 0;
    std::string stencilScheme = "CD2";
    std::string reason;
    CellExpr withResource(const Resource &r) const {
        auto e = *this;
        e.reference.info.read(&r);
        return e;
    }
    template <class S> Term fusedTerm() const {
        auto typed = detail::Compile<Node, S>::get(node);
        Term t = Fused::fuse(typed, *reference.field);
        t.info = reference.info;
        t.info.execution = "fused";
        t.inputs = reference.inputs;
        t.ghostRequired = reference.ghostRequired;
        return t;
    }
    Term lower(Fusion mode) const {
        const bool available = Fusible && stencils <= 1 && reason.empty();
        if (mode == Fusion::Required && !available)
            throw std::invalid_argument(
                "required fusion unavailable: " +
                (reason.empty() ? "multiple stencils require materialization" : reason));
        if constexpr (Fusible) {
            if (mode != Fusion::Off && available) {
                if (stencilScheme == "CD2")
                    return fusedTerm<scheme::CD2>();
                if (stencilScheme == "CD4")
                    return fusedTerm<scheme::CD4>();
                if (stencilScheme == "CD6")
                    return fusedTerm<scheme::CD6>();
                if (stencilScheme == "Iso9")
                    return fusedTerm<scheme::Iso9>();
                if (mode == Fusion::Required)
                    throw std::invalid_argument("required fusion: unsupported scheme " +
                                                stencilScheme);
            }
        }
        Term t = reference;
        t.info.execution =
            "materialized: " + (mode == Fusion::Off
                                    ? std::string("fusion off")
                                    : (reason.empty() ? "multiple stencil region" : reason));
        return t;
    }
};
template <class N, bool F> auto operator*(CellExpr<N, F> a, double c) {
    return CellExpr<Fused::ScaleNode<N>, F>{
        {a.node, Real(c)}, a.reference * c, a.stencils, a.stencilScheme, a.reason};
}
template <class N, bool F> auto operator*(double c, CellExpr<N, F> a) {
    return a * c;
}
template <class N, bool F> auto operator-(CellExpr<N, F> a) {
    return a * (-1);
}
template <class N, bool F> auto operator/(CellExpr<N, F> a, double c) {
    return a * (1 / c);
}
template <class A, bool FA, class B, bool FB> auto operator+(CellExpr<A, FA> a, CellExpr<B, FB> b) {
    return CellExpr < Fused::AddNode<A, B>,
           FA && FB > {{a.node, b.node},
                       sum(a.reference, b.reference),
                       a.stencils + b.stencils,
                       a.stencils ? a.stencilScheme : b.stencilScheme,
                       a.reason.empty() ? b.reason : a.reason};
}
template <class A, bool FA, class B, bool FB> auto operator-(CellExpr<A, FA> a, CellExpr<B, FB> b) {
    return a + (-b);
}
template <class A, bool FA, class B, bool FB> auto operator*(CellExpr<A, FA> a, CellExpr<B, FB> b) {
    return CellExpr < Fused::MulNode<A, B>,
           FA && FB > {{a.node, b.node},
                       PhiX::mul(a.reference, b.reference),
                       a.stencils + b.stencils,
                       a.stencils ? a.stencilScheme : b.stencilScheme,
                       a.reason.empty() ? b.reason : a.reason};
}
template <class N, bool F, class Fn> auto pw(CellExpr<N, F> a, Pure<Fn> f) {
    return CellExpr<detail::Transform<N, Fn>, F>{
        {a.node, f.fn}, PhiX::pw(a.reference, f.fn), a.stencils, a.stencilScheme, a.reason};
}
inline auto adapted(Term t) {
    return CellExpr<detail::Opaque, false>{
        {}, std::move(t), 0, "CD2", "precompiled launcher: no expression fusion candidate"};
}
inline auto adapted(const RHSExpr &e) {
    return adapted(collapse(e));
}
inline auto adapted(const ExprTree &e, const BcMap &bcs = {}) {
    return adapted(collapse(lowerExprTree(e, bcs).expression()));
}

class Spatial {
    const Schemes &schemes_;
    static void check2D(const ScalarField &f) {
        if (f.mesh.dim != 2)
            throw std::invalid_argument("numerics::Spatial supports 2D only");
        check::checkCartesian(f.mesh, "Spatial");
    }
    Schemes::Selection select(const char *family, const char *op, const ScalarField &f,
                              const std::string &key) const {
        check2D(f);
        return key.empty() ? schemes_.select(family, std::string(op) + "(" + f.name + ")")
                           : schemes_.named(family, key);
    }

  public:
    explicit Spatial(const Schemes &s) : schemes_(s) {}
    CellExpr<Fused::FieldNode> value(const ScalarField &f) const;

    template <class Fn> auto pw(const ScalarField &f, Pure<Fn> fn) const {
        check2D(f);
        return CellExpr<Fused::Pw1Node<Fn>>{Fused::fpw(f, fn.fn), PhiX::pw(f, fn.fn)};
    }
    template <class Fn> auto pw(const ScalarField &a, const ScalarField &b, Pure<Fn> fn) const {
        check2D(a);
        check::checkSameMesh(a, b, "Spatial::pw");
        return CellExpr<Fused::Pw2Node<Fn>>{Fused::fpw2(a, b, fn.fn), PhiX::pw(a, b, fn.fn)};
    }
    template <class Fn>
    auto pw(const ScalarField &a, const ScalarField &b, const ScalarField &c, Pure<Fn> fn) const {
        check2D(a);
        check::checkSameMesh(a, b, "Spatial::pw");
        check::checkSameMesh(a, c, "Spatial::pw");
        return CellExpr<Fused::Pw3Node<Fn>>{Fused::fpw3(a, b, c, fn.fn), PhiX::pw(a, b, c, fn.fn)};
    }
    template <class Fn>
    auto pw(const ScalarField &a, const ScalarField &b, const ScalarField &c, const ScalarField &d,
            Pure<Fn> fn) const {
        check2D(a);
        check::checkSameMesh(a, b, "Spatial::pw");
        check::checkSameMesh(a, c, "Spatial::pw");
        check::checkSameMesh(a, d, "Spatial::pw");
        return CellExpr<Fused::Pw4Node<Fn>>{Fused::fpw4(a, b, c, d, fn.fn),
                                            PhiX::pw(a, b, c, d, fn.fn)};
    }
    // Functor-first form supports all 1..8 arities of the ordinary pw backend.
    template <class Fn, class... Fields>
    auto pw(Pure<Fn> fn, const ScalarField &a, const Fields &...rest) const {
        static_assert(sizeof...(Fields) < 8, "Spatial::pw supports 1..8 fields");
        check2D(a);
        (check::checkSameMesh(a, rest, "Spatial::pw"), ...);
        using Node = detail::Pointwise<Fn, 1 + sizeof...(Fields)>;
        return CellExpr<Node>{{fn.fn, {&a, &rest...}}, PhiX::pw(a, rest..., fn.fn)};
    }
    auto lap(const ScalarField &f, const std::string &key = "") const {
        auto sel = select("laplacian", "lap", f, key);
        auto t = PhiX::lap(f, sel.name);
        t.info.schemes = {sel.describe()};
        return CellExpr<detail::Stencil<true>>{
            {&f, 0},
            t,
            1,
            sel.name,
            sel.name == "Iso27" ? "legacy Iso27 uses CD2 in 2D; select CD2 explicitly for fusion"
                                : ""};
    }
    auto grad(const ScalarField &f, int axis, const std::string &key = "") const {
        if (axis < 0 || axis > 1)
            throw std::invalid_argument("Spatial::grad axis must be x or y");
        check2D(f);
        auto sel = key.empty() ? schemes_.gradient(f.name, axis) : schemes_.named("gradient", key);
        auto t = PhiX::grad(f, axis, sel.name);
        t.info.schemes = {sel.describe()};
        return CellExpr<detail::Stencil<false>>{{&f, axis}, t, 1, sel.name};
    }
    auto gradDot(const ScalarField &f, const ScalarField &g) const {
        check2D(f);
        check2D(g);
        return adapted(PhiX::grad_dot(f, g, schemes_));
    }
    auto gradSq(const ScalarField &f, const std::string &key = "") const {
        auto sel = select("gradSq", "gradSq", f, key);
        auto t = PhiX::gradSq(f, sel.name);
        t.info.schemes = {sel.describe()};
        return adapted(t);
    }
    auto adv(const VectorField &u, const ScalarField &f, const std::string &key = "") const {
        check2D(f);
        auto sel =
            key.empty() ? schemes_.advection(u.name, f.name) : schemes_.named("advection", key);
        auto t = PhiX::adv(u, f, sel.name);
        t.info.schemes = {sel.describe()};
        return adapted(t);
    }
    auto anisoDiv(const ScalarField &f, AnisoParams p, const std::string &key = "") const {
        auto sel = select("anisoDiv", "anisoDiv", f, key);
        p.scheme = anisoSchemeFromString(sel.name);
        auto t = PhiX::anisoDiv(f, p);
        t.info.schemes = {sel.describe()};
        return adapted(t);
    }
    auto anisoDiv(const ScalarField &f, const ScalarField &theta, AnisoParams p,
                  const std::string &key = "") const {
        auto sel = select("anisoDiv", "anisoDiv", f, key);
        p.scheme = anisoSchemeFromString(sel.name);
        auto t = PhiX::anisoDiv(f, theta, p);
        t.info.schemes = {sel.describe()};
        return adapted(t);
    }
};
} // namespace PhiX::numerics
