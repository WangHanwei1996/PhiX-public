#pragma once
#include "numerics/System.h"
#include "operators/FaceOps.h"

namespace PhiX::numerics {
namespace face_detail {
struct Context {
    int sx, sy, g, nx, ny;
    Real dx, dy;
};
__host__ __device__ inline int cell(int i, int j, const Context &c) {
    return i + c.g + c.sx * (j + c.g + c.sy * c.g);
}
__host__ __device__ inline int face(int i, int j, int axis, int sx, int sy, int g) {
    return (axis == 0 ? i : i + g) + sx * ((axis == 1 ? j : j + g) + sy * g);
}
template <int Axis, int Kind> struct CellNode {
    const ScalarField *field;
    const Real *data = nullptr;
    bool nearest = false;
    int component = Axis;
    auto bound(bool gpu) const {
        auto n = *this;
        n.data = Fused::detail::current(field, gpu, data);
        return n;
    }
    __host__ __device__ Real eval(int i, int j, const Context &c) const {
        int hi = cell(i, j, c), stride = Axis == 0 ? 1 : c.sx, lo = hi - stride;
        if constexpr (Kind == 0) {
            if (nearest && (Axis == 0 ? i : j) == 0)
                return data[hi];
            if (nearest && (Axis == 0 ? i : j) == (Axis == 0 ? c.nx : c.ny))
                return data[lo];
            return Real(0.5) * (data[lo] + data[hi]);
        } else {
            if (component == Axis)
                return (data[hi] - data[lo]) / (Axis == 0 ? c.dx : c.dy);
            const int tangent = Axis == 0 ? c.sx : 1;
            return Real(0.25) *
                   (data[hi + tangent] - data[hi - tangent] + data[lo + tangent] -
                    data[lo - tangent]) /
                   (Axis == 0 ? c.dy : c.dx);
        }
    }
};
template <int Axis> struct FaceNode {
    const FaceField *field;
    const Real *data = nullptr;
    int sx, sy, g;
    auto bound(bool gpu) const {
        auto n = *this;
        if (gpu && !field->d_data)
            throw std::invalid_argument("face expression: missing device input");
        n.data = gpu ? field->d_data : field->data.data();
        return n;
    }
    __host__ __device__ Real eval(int i, int j, const Context &) const {
        return data[face(i, j, Axis, sx, sy, g)];
    }
};
template <class A, class B, bool Product> struct Binary {
    A a;
    B b;
    auto bound(bool gpu) const {
        return Binary<decltype(a.bound(gpu)), decltype(b.bound(gpu)), Product>{a.bound(gpu),
                                                                               b.bound(gpu)};
    }
    __host__ __device__ Real eval(int i, int j, const Context &c) const {
        if constexpr (Product)
            return a.eval(i, j, c) * b.eval(i, j, c);
        else
            return a.eval(i, j, c) + b.eval(i, j, c);
    }
};
template <class N, class Fn> struct Transform {
    N n;
    Fn fn;
    auto bound(bool gpu) const { return Transform<decltype(n.bound(gpu)), Fn>{n.bound(gpu), fn}; }
    __host__ __device__ Real eval(int i, int j, const Context &c) const {
        return fn(n.eval(i, j, c));
    }
};
template <class A, class B, class C, class Fn> struct Transform3 {
    A a;
    B b;
    C c;
    Fn fn;
    auto bound(bool gpu) const {
        return Transform3<decltype(a.bound(gpu)), decltype(b.bound(gpu)),
                          decltype(c.bound(gpu)), Fn>{a.bound(gpu), b.bound(gpu), c.bound(gpu), fn};
    }
    __host__ __device__ Real eval(int i, int j, const Context &ctx) const {
        return fn(a.eval(i, j, ctx), b.eval(i, j, ctx), c.eval(i, j, ctx));
    }
};
struct Scale {
    Real c;
    __host__ __device__ Real operator()(Real x) const { return c * x; }
};
template <int Axis, class N>
__global__ void writeFaces(Real *out, N node, Context c, int sx, int sy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, nx = c.nx + (Axis == 0),
        ny = c.ny + (Axis == 1);
    if (tid >= nx * ny)
        return;
    int i = tid % nx, j = tid / nx;
    out[face(i, j, Axis, sx, sy, c.g)] = node.eval(i, j, c);
}
// Materialized face values live in the caller's persistent ScratchPool. Rebuilding
// an expression (for changed material parameters) never owns/frees device buffers.
template <int Axis> struct RawNode {
    const Real *data;
    int sx, sy, g;
    RawNode bound(bool) const { return *this; }
    __host__ __device__ Real eval(int i, int j, const Context &) const {
        return data[face(i, j, Axis, sx, sy, g)];
    }
};
template <int Axis> auto raw(const Real *data, const ScalarField &f) {
    return RawNode<Axis>{data, f.mesh.n[0] + (Axis == 0 ? 1 : 2 * f.ghost),
                         f.mesh.n[1] + (Axis == 1 ? 1 : 2 * f.ghost), f.ghost};
}
template <int Axis> Real *acquire(const ScalarField &f, Backend b, ScratchPool &pool) {
    auto n = raw<Axis>(nullptr, f);
    const auto size = std::size_t(n.sx) * n.sy * (1 + 2 * f.ghost);
    return b == Backend::CUDA ? pool.acquireDevice(size) : pool.acquireHost(size);
}
template <int Axis, class N>
void write(Real *out, N node, const ScalarField &layout, Backend backend, ScratchPool &pool) {
    Context c{layout.storedDims[0], layout.storedDims[1],   layout.ghost,          layout.mesh.n[0],
              layout.mesh.n[1],     Real(layout.mesh.d[0]), Real(layout.mesh.d[1])};
    auto n = node.bound(backend == Backend::CUDA);
    auto shape = raw<Axis>(out, layout);
    int nx = c.nx + (Axis == 0), ny = c.ny + (Axis == 1);
    if (!out)
        throw std::invalid_argument("face expression: output not allocated");
    if (backend == Backend::CUDA) {
        writeFaces<Axis>
            <<<(nx * ny + 255) / 256, 256, 0, pool.stream>>>(out, n, c, shape.sx, shape.sy);
        PHIX_KERNEL_CHECK("face expression");
    } else
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                out[face(i, j, Axis, shape.sx, shape.sy, c.g)] = n.eval(i, j, c);
}
inline void layout(const FaceField &face, const ScalarField &cell, int axis) {
    if (face.normalAxis != axis || face.ghost != cell.ghost ||
        !check::sameMeshGeometry(face.mesh, cell.mesh))
        throw std::invalid_argument("face expression: output position/layout mismatch");
}
} // namespace face_detail

template <int Axis, class Node> struct FaceExpr {
    static_assert(Axis == 0 || Axis == 1, "2D face axis must be x or y");
    static constexpr int axis = Axis;
    Node node;
    const ScalarField *layout;
    OperationInfo info;
    bool certified = true;
    std::function<void(Real *, Backend, ScratchPool &)> reference;
    void evaluate(FaceField &out, Backend backend, Fusion fusion, ScratchPool &pool) const {
        face_detail::layout(out, *layout, Axis);
        Real *data = backend == Backend::CUDA ? out.d_data : out.data.data();
        if (fusion == Fusion::Off)
            reference(data, backend, pool);
        else
            face_detail::write<Axis>(data, node, *layout, backend, pool);
    }
    void evaluate(FaceField &out, Backend backend, Fusion fusion) const {
        ScratchPool pool;
        evaluate(out, backend, fusion, pool);
    }
};
template <int Axis, class N> auto faceValue(const FaceField &f, const FaceExpr<Axis, N> &layout) {
    face_detail::layout(f, *layout.layout, Axis);
    auto n = face_detail::FaceNode<Axis>{&f, nullptr, f.storedDims[0], f.storedDims[1], f.ghost};
    OperationInfo info;
    info.complete = true;
    info.read(&f);
    const auto *cell = layout.layout;
    return FaceExpr<Axis, decltype(n)>{n, cell, info, true,
                                       [=](Real *out, Backend b, ScratchPool &pool) {
                                           face_detail::write<Axis>(out, n, *cell, b, pool);
                                       }};
}
template <bool Product, int Axis, class A, class B>
auto faceBinary(FaceExpr<Axis, A> a, FaceExpr<Axis, B> b) {
    check::checkSameMesh(*a.layout, *b.layout, "face binary expression");
    auto node = face_detail::Binary<A, B, Product>{a.node, b.node};
    auto info = a.info;
    info.merge(b.info);
    auto ref = [=](Real *out, Backend backend, ScratchPool &pool) {
        auto *x = face_detail::acquire<Axis>(*a.layout, backend, pool);
        auto *y = face_detail::acquire<Axis>(*a.layout, backend, pool);
        a.reference(x, backend, pool);
        b.reference(y, backend, pool);
        auto l = face_detail::raw<Axis>(x, *a.layout);
        auto r = face_detail::raw<Axis>(y, *a.layout);
        face_detail::write<Axis>(out, face_detail::Binary<decltype(l), decltype(r), Product>{l, r},
                                 *a.layout, backend, pool);
    };
    return FaceExpr<Axis, decltype(node)>{node, a.layout, info, a.certified && b.certified, ref};
}
template <int A, class L, class R> auto operator*(FaceExpr<A, L> l, FaceExpr<A, R> r) {
    return faceBinary<true>(l, r);
}
template <int A, class L, class R> auto operator+(FaceExpr<A, L> l, FaceExpr<A, R> r) {
    return faceBinary<false>(l, r);
}
template <int Axis, class N, class Fn> auto pw(FaceExpr<Axis, N> a, Pure<Fn> f) {
    auto node = face_detail::Transform<N, Fn>{a.node, f.fn};
    auto ref = [=](Real *out, Backend b, ScratchPool &pool) {
        auto *temp = face_detail::acquire<Axis>(*a.layout, b, pool);
        a.reference(temp, b, pool);
        auto leaf = face_detail::raw<Axis>(temp, *a.layout);
        face_detail::write<Axis>(out, face_detail::Transform<decltype(leaf), Fn>{leaf, f.fn},
                                 *a.layout, b, pool);
    };
    return FaceExpr<Axis, decltype(node)>{node, a.layout, a.info, a.certified, ref};
}
// Multi-input face algebra: all arguments refer to the same oriented face.
template <int Axis, class A, class B, class C, class Fn>
auto pw(FaceExpr<Axis, A> a, FaceExpr<Axis, B> b, FaceExpr<Axis, C> c, Pure<Fn> fn) {
    check::checkSameMesh(*a.layout, *b.layout, "face pw");
    check::checkSameMesh(*a.layout, *c.layout, "face pw");
    auto node = face_detail::Transform3<A, B, C, Fn>{a.node, b.node, c.node, fn.fn};
    auto info = a.info;
    info.merge(b.info);
    info.merge(c.info);
    auto ref = [=](Real *out, Backend backend, ScratchPool &pool) {
        auto *x = face_detail::acquire<Axis>(*a.layout, backend, pool);
        auto *y = face_detail::acquire<Axis>(*a.layout, backend, pool);
        auto *z = face_detail::acquire<Axis>(*a.layout, backend, pool);
        a.reference(x, backend, pool);
        b.reference(y, backend, pool);
        c.reference(z, backend, pool);
        auto l = face_detail::raw<Axis>(x, *a.layout);
        auto m = face_detail::raw<Axis>(y, *a.layout);
        auto r = face_detail::raw<Axis>(z, *a.layout);
        face_detail::write<Axis>(out,
            face_detail::Transform3<decltype(l), decltype(m), decltype(r), Fn>{l, m, r, fn.fn},
            *a.layout, backend, pool);
    };
    return FaceExpr<Axis, decltype(node)>{node, a.layout, info,
                                        a.certified && b.certified && c.certified, ref};
}
template <int A, class N> auto operator*(FaceExpr<A, N> n, double c) {
    return pw(n, pure(face_detail::Scale{Real(c)}));
}
template <int A, class N> auto operator*(double c, FaceExpr<A, N> n) {
    return n * c;
}
template <int A, class N> auto operator-(FaceExpr<A, N> n) {
    return n * (-1);
}

template <int A, class N> void defineFace(System &system, FaceField &out, FaceExpr<A, N> expr) {
    face_detail::layout(out, *expr.layout, A);
    auto mode = std::make_shared<Fusion>(Fusion::Auto);
    auto pool = std::make_shared<ScratchPool>();
    ExternalStep step;
    step.name = out.name;
    step.reads = expr.info;
    step.faceWrites = {&out};
    step.configure = [=](Backend, Fusion f) { *mode = f; };
    step.describe = [=] {
        return *mode == Fusion::Off ? "materialized face producer" : "fused face producer";
    };
    step.cpu = [&out, expr, mode, pool](double, double) {
        pool->reset();
        expr.evaluate(out, Backend::CPU, *mode, *pool);
    };
    step.gpu = [&out, expr, mode, pool](double, double) {
        pool->reset();
        expr.evaluate(out, Backend::CUDA, *mode, *pool);
    };
    system.producer(std::move(step));
}

namespace face_detail {
struct AllCells {
    AllCells bound(bool) const { return *this; }
    void describe(OperationInfo&, const ScalarField&) const {}
    __host__ __device__ bool active(int, int, Context) const { return true; }
};
template <class Fn> struct CellPredicate {
    const ScalarField* field;
    Fn fn;
    const Real* data = nullptr;
    CellPredicate bound(bool gpu) const {
        auto copy = *this;
        copy.data = gpu ? field->d_curr : field->curr.data();
        if (!copy.data) throw std::runtime_error("flux cell predicate: field not allocated");
        return copy;
    }
    void describe(OperationInfo& info, const ScalarField& layout) const {
        check::checkSameMesh(layout, *field, "flux cell predicate");
        if (field->ghost != layout.ghost || field->storedSize != layout.storedSize)
            throw std::invalid_argument("flux cell predicate: layout mismatch");
        info.read(field);
    }
    __host__ __device__ bool active(int i, int j, Context c) const {
        return fn(data[cell(i,j,c)]);
    }
};
template <class X, class Y> __host__ __device__ Real divergence(X x, Y y, int i, int j, Context c) {
    return (x.eval(i + 1, j, c) - x.eval(i, j, c)) / c.dx +
           (y.eval(i, j + 1, c) - y.eval(i, j, c)) / c.dy;
}
template <class X, class Y, class Gate>
__global__ void assembled(Real *out, X x, Y y, Context c, Real coeff, Gate gate) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c.nx * c.ny)
        return;
    int i = tid % c.nx, j = tid / c.nx;
    if (gate.active(i,j,c))
        out[cell(i, j, c)] += coeff * divergence(x, y, i, j, c);
}
} // namespace face_detail
template <class X, class Y> struct Flux {
    std::string name;
    X x;
    Y y;
};
template <class X, class Y, class Gate = face_detail::AllCells> struct FluxDivergence {
    Flux<X, Y> flux;
    double coeff = 1;
    Gate gate{};
    // Select the receiving-cell RHS after divergence, not the face flux itself.
    // Auto skips face evaluation in inactive cells; Off materializes faces first.
    template <class Fn> auto where(const ScalarField& field, Pure<Fn> predicate) const {
        static_assert(std::is_same_v<Gate, face_detail::AllCells>,
                      "combine predicates explicitly instead of replacing an existing where");
        return FluxDivergence<X,Y,face_detail::CellPredicate<Fn>>{
            flux, coeff, {&field, predicate.fn}};
    }
    Term lower(Fusion mode) const {
        const auto x = flux.x;
        const auto y = flux.y;
        const auto *layout = x.layout;
        check::checkSameMesh(*layout, *y.layout, "flux divergence");
        if (mode == Fusion::Required && (!x.certified || !y.certified))
            throw std::invalid_argument(
                "flux fusion requires certified reconstruction (LinearGhost/CD2)");
        Term t;
        t.type = TermType::COMPOSITE;
        t.field = layout;
        t.rhsGhost = layout->ghost;
        t.coeff = coeff;
        t.info = x.info;
        t.info.merge(y.info);
        gate.describe(t.info, *layout);
        const auto selection = gate;
        for (const auto &r : t.info.reads)
            if (r.cell) {
                t.inputs.push_back(r.cell);
                t.ghostRequired = std::max(t.ghostRequired, r.halo);
            }
        const face_detail::Context context{layout->storedDims[0],  layout->storedDims[1],
                                           layout->ghost,          layout->mesh.n[0],
                                           layout->mesh.n[1],      Real(layout->mesh.d[0]),
                                           Real(layout->mesh.d[1])};
        if (mode == Fusion::Off || !x.certified || !y.certified) {
            t.gpu_launcher = [=](Real *rhs, double c, ScratchPool &pool) {
                auto *fx = face_detail::acquire<0>(*layout, Backend::CUDA, pool);
                auto *fy = face_detail::acquire<1>(*layout, Backend::CUDA, pool);
                x.reference(fx, Backend::CUDA, pool);
                y.reference(fy, Backend::CUDA, pool);
                face_detail::
                    assembled<<<(context.nx * context.ny + 255) / 256, 256, 0, pool.stream>>>(
                        rhs, face_detail::raw<0>(fx, *layout), face_detail::raw<1>(fy, *layout),
                        context, Real(c), selection.bound(true));
                PHIX_KERNEL_CHECK("materialized face divergence");
            };
            t.cpu_kernel = [=](Real *rhs, double c, ScratchPool &pool) {
                auto *fx = face_detail::acquire<0>(*layout, Backend::CPU, pool);
                auto *fy = face_detail::acquire<1>(*layout, Backend::CPU, pool);
                x.reference(fx, Backend::CPU, pool);
                y.reference(fy, Backend::CPU, pool);
                const auto active = selection.bound(false);
                for (int j = 0; j < context.ny; ++j)
                    for (int i = 0; i < context.nx; ++i)
                      if (active.active(i,j,context))
                        rhs[face_detail::cell(i, j, context)] +=
                            Real(c) * face_detail::divergence(face_detail::raw<0>(fx, *layout),
                                                              face_detail::raw<1>(fy, *layout), i,
                                                              j, context);
            };
            t.info.execution = "materialized face chain: " + flux.name;
        } else {
            t.gpu_launcher = [=](Real *rhs, double c, ScratchPool &pool) {
                face_detail::
                    assembled<<<(context.nx * context.ny + 255) / 256, 256, 0, pool.stream>>>(
                        rhs, x.node.bound(true), y.node.bound(true), context, Real(c), selection.bound(true));
                PHIX_KERNEL_CHECK("assembled flux expression");
            };
            t.cpu_kernel = [=](Real *rhs, double c, ScratchPool &) {
                auto bx = x.node.bound(false);
                auto by = y.node.bound(false);
                const auto active = selection.bound(false);
                for (int j = 0; j < context.ny; ++j)
                    for (int i = 0; i < context.nx; ++i)
                      if (active.active(i,j,context))
                        rhs[face_detail::cell(i, j, context)] +=
                            Real(c) * face_detail::divergence(bx, by, i, j, context);
            };
            t.info.execution =
                "assembled flux: " + flux.name + " (identical oriented face evaluations)";
        }
        return t;
    }
};
template <class X, class Y, class G> auto operator*(FluxDivergence<X, Y, G> d, double c) {
    d.coeff *= c;
    return d;
}
template <class X, class Y, class G> auto operator*(double c, FluxDivergence<X, Y, G> d) {
    return d * c;
}
template <class X, class Y, class G> auto operator-(FluxDivergence<X, Y, G> d) {
    return d * (-1);
}

class Faces {
    const Schemes &schemes_;
    Schemes::Selection selected(const char *family, const std::string &key) const {
        return schemes_.select(family, key);
    }
    template <int Axis, int Kind>
    auto reconstruction(const ScalarField &f, int component, const char *family,
                        const std::string &key) const {
        if (f.mesh.dim != 2)
            throw std::invalid_argument("face expressions support 2D only");
        auto s = selected(family, key);
        auto fam = scheme::findFamily(family)->id;
        auto entry = scheme::detail::lookup(scheme::detail::faceFactories(fam), s.name,
                                            "face reconstruction", false);
        const bool nearest = entry.factory() == scheme::FaceMethod::NearestCell;
        check::checkGhost(f, entry.descriptor.primaryHalo, "face reconstruction");
        auto node = face_detail::CellNode<Axis, Kind>{&f, nullptr, nearest, component};
        OperationInfo info;
        info.complete = true;
        info.read(&f, entry.descriptor.primaryHalo, component != Axis);
        info.schemes = {s.describe()};
        return FaceExpr<Axis, decltype(node)>{node, &f, info, !nearest,
                                              [=, &f](Real *out, Backend b, ScratchPool &pool) {
                                                  face_detail::write<Axis>(out, node, f, b, pool);
                                              }};
    }

  public:
    explicit Faces(const Schemes &s) : schemes_(s) {}
    template <int A> auto interp(const ScalarField &f, std::integral_constant<int, A>) const {
        return reconstruction<A, 0>(f, A, "interpolation", "interp(" + f.name + ")");
    }
    template <int A> auto snGrad(const ScalarField &f, std::integral_constant<int, A>) const {
        return reconstruction<A, 1>(f, A, "snGrad", "snGrad(" + f.name + ")");
    }
    template <int A>
    auto gradComponent(const ScalarField &f, std::integral_constant<int, A>, int component) const {
        if (component < 0 || component > 1)
            throw std::invalid_argument("face gradient component must be x or y");
        return reconstruction<A, 1>(f, component, "faceGrad", "faceGrad(" + f.name + ")");
    }
    template <class Fn> auto flux(std::string name, Fn fn) const {
        auto x = fn(std::integral_constant<int, 0>{});
        auto y = fn(std::integral_constant<int, 1>{});
        return Flux<decltype(x), decltype(y)>{std::move(name), x, y};
    }
    template <class X, class Y> auto div(Flux<X, Y> flux) const {
        auto s = selected("fluxDiv", "div(" + flux.name + ")");
        scheme::detail::lookup(scheme::detail::faceFactories(scheme::Family::FluxDiv), s.name,
                               "flux divergence", false)
            .factory();
        flux.x.info.schemes.push_back(s.describe());
        return FluxDivergence<X, Y>{flux};
    }
};
} // namespace PhiX::numerics
