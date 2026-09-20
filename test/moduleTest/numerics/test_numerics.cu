#include "numerics/Face.h"
#include "numerics/Multi.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
using namespace PhiX;
namespace num = PhiX::numerics;
using num::pure;
namespace {
constexpr double pi = 3.141592653589793;
constexpr double tolerance = std::is_same<Real, float>::value ? 3e-3 : 3e-10;
int checks = 0;
void require(bool ok, const char *what) {
    ++checks;
    if (!ok)
        throw std::runtime_error(what);
}
template <class Fn> void rejects(Fn fn, const char *what) {
    bool bad = false;
    try {
        fn();
    } catch (const std::exception &) {
        bad = true;
    }
    require(bad, what);
}
struct Identity {
    __host__ __device__ Real operator()(Real x) const { return x; }
};
struct Nonlinear {
    __host__ __device__ Real operator()(Real x) const { return x * x * x - x; }
};
struct Square {
    __host__ __device__ Real operator()(Real x) const { return x * x; }
};
Mesh mesh() {
    return Mesh::makeUniform2D(CoordSys::CARTESIAN, 24, 1.0 / 24, 0, 20, 1.0 / 24, 0);
}
void init(ScalarField &f) {
    f.initialize([](double x, double y, double) {
        return .2 + .05 * cos(2 * pi * x) + .03 * sin(2 * pi * y / (20. / 24));
    });
}
void device(ScalarField &f) {
    f.allocDevice();
    f.uploadAllToDevice();
}
double diff(const ScalarField &a, const ScalarField &b) {
    double e = 0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
        for (int i = 0; i < a.mesh.n[0]; ++i)
            e = std::max(e,
                         std::abs(double(a.curr[a.index(i, j)]) - double(b.curr[b.index(i, j)])));
    return e;
}
void periodic(ScalarField &f) {
    int nx = f.mesh.n[0], ny = f.mesh.n[1], g = f.ghost;
    for (int j = -g; j < ny + g; ++j)
        for (int i = -g; i < nx + g; ++i)
            if (i < 0 || i >= nx || j < 0 || j >= ny)
                f.curr[f.index(i, j)] = f.curr[f.index((i + nx) % nx, (j + ny) % ny)];
}
void evaluate(Term t, ScalarField &out, bool gpu) {
    Equation eq(out);
    eq.setRHS(t);
    if (gpu) {
        eq.computeRHS(out);
        out.downloadCurrFromDevice();
    } else
        eq.computeRHSCPU(out);
}
double mass(const ScalarField &f) {
    double sum = 0;
    for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < f.mesh.n[0]; ++i)
            sum += f.curr[f.index(i, j)];
    return sum;
}
void config() {
    auto s = Schemes::fromJson({{"schema", 2},
                                {"policy", "strict"},
                                {"laplacian", {{"default", "none"}, {"capillary", "Iso9"}}},
                                {"advection", {{"adv(u,c)", "UW2"}}}});
    require(s.named("laplacian", "capillary").name == "Iso9", "named selection");
    auto directional =
        Schemes::fromJson({{"gradient", {{"grad(c)", "CD2"}, {"grad(c,x)", "CD4"}}}});
    require(directional.gradient("c", 0).name == "CD4" &&
                directional.gradient("c", 1).name == "CD2",
            "axis-specific selection then field fallback");
    require(s.advection("u", "c").name == "UW2", "full transport key");
    rejects([&] { s.lap("c"); }, "default none must reject");
    rejects([&] { s.grad("c"); }, "strict missing family must reject");
    rejects([&] { s.named("laplacian", "capilary"); }, "named typo must reject");
    rejects([] { Schemes::fromJson({{"policy", "strict"}}); }, "policy requires schema");
    auto old = Schemes::fromJson({{"advection", {{"adv(c)", "UW1"}}}});
    require(old.advection("u", "c").matchedKey == "adv(c)", "legacy transport key provenance");
}
void fragments() {
    auto m = mesh();
    ScalarField f(m, "f", 3), g(m, "g", 3), cpu(m, "cpu", 3), off(m, "off", 3), on(m, "on", 3);
    init(f);
    init(g);
    periodic(f);
    periodic(g);
    device(f);
    device(g);
    device(off);
    device(on);
    for (const char *name : {"CD2", "CD4", "CD6", "Iso9"}) {
        auto sch = Schemes::builtin({{"laplacian", name}, {"gradient", name}});
        num::Spatial ops(sch);
        auto expr = ops.pw(f, pure(Nonlinear{})) - .02 * ops.lap(f);
        evaluate(expr.lower(num::Fusion::Off), cpu, false);
        evaluate(expr.lower(num::Fusion::Off), off, true);
        evaluate(expr.lower(num::Fusion::Required), on, true);
        require(diff(cpu, off) < tolerance, "ordinary CPU/CUDA equality");
        require(diff(off, on) < tolerance, "one-stencil fused equality");
        Equation treeEq(f);
        treeEq.setRHS(expr_lap(f, sch));
        treeEq.computeRHSCPU(cpu);
        evaluate(PhiX::lap(f, std::string(name)), off, false);
        require(diff(cpu, off) < tolerance, "ExprTree preserves configured stencil");
        PeriodicBC px(f.mesh.patch("xmin")), py(f.mesh.patch("ymin"));
        evaluate(PhiX::lap(PhiX::pw(f, Identity{}), {&px, &py}, std::string(name)), cpu, false);
        require(diff(cpu, off) < tolerance,
                "materialized composite preserves configured stencil including corners");
        auto gradient = ops.value(g) * ops.grad(f, 1);
        evaluate(gradient.lower(num::Fusion::Off), off, true);
        evaluate(gradient.lower(num::Fusion::Required), on, true);
        require(diff(off, on) < tolerance, "gradient fused selection retained");
        auto multi = ops.lap(f) + ops.lap(g);
        rejects([&] { multi.lower(num::Fusion::Required); },
                "multiple stencil required fusion must fail");
        require(multi.lower(num::Fusion::Auto).info.execution.find("materialized") !=
                    std::string::npos,
                "auto fallback reported");
    }
    auto sch = Schemes::builtin();
    num::Spatial ops(sch);
    ScalarField unallocated(m, "late", 3);
    unallocated.fill(2);
    Term fused = Fused::fuse(Fused::ffield(unallocated), unallocated);
    evaluate(fused, cpu, false);
    require(std::abs(cpu.curr[cpu.index(2, 2)] - 2) < tolerance,
            "CPU construction before allocation");
    device(unallocated);
    ScalarField replacement(m, "new", 3);
    replacement.fill(7);
    device(replacement);
    std::swap(unallocated.d_curr, replacement.d_curr);
    evaluate(fused, on, true);
    require(std::abs(on.curr[on.index(2, 2)] - 7) < tolerance, "fused current pointer rebind");
    std::swap(unallocated.d_curr, replacement.d_curr);
    auto shell = ScalarField::makeShell(f.mesh, f.ghost, f.d_curr);
    Term copy = Fused::fuse(Fused::ffield(f), f);
    Equation alias(f);
    alias.setRHS(copy);
    alias.computeRHS(shell);
    f.downloadCurrFromDevice();
    require(std::abs(f.curr[f.index(0, 0)] - .2) > .001, "raw shell alias retains input values");
    periodic(f);
    f.uploadAllToDevice();
    // Target is the second operand, not the representative field.
    Term composite = PhiX::mul(PhiX::pw(f, Identity{}), PhiX::pw(g, Identity{}));
    evaluate(composite, cpu, false);
    evaluate(composite, g, false);
    require(diff(cpu, g) < tolerance, "composite alias on secondary input");
    init(g);
    periodic(g);
    g.uploadAllToDevice();
    evaluate(composite, g, true);
    require(diff(cpu, g) < tolerance, "GPU composite alias on secondary input");
    init(g);
    auto tree = ExprTree(f) * expr_pw(g, Square{}, 2);
    Equation e(g);
    e.setRHS(tree);
    e.computeRHSCPU(cpu);
    e.computeRHSCPU(g);
    require(diff(cpu, g) < tolerance, "ExprTree pointwise input metadata survives lowering");
    BcMap bcs;
    PeriodicBC px(m.patch("xmin")), py(m.patch("ymin"));
    bcs[&f] = {&px, &py};
    ExprTree child = 2 * (ExprTree(f) + ExprTree(f));
    ExprTree nested(std::make_shared<ExprStencil>(StencilKind::LAP, 0, 1, child.node));
    auto lowered = lowerExprTree(nested, bcs).expression();
    evaluate(num::collapse(lowered), cpu, false);
    evaluate(4 * PhiX::lap(f), off, false);
    require(diff(cpu, off) < tolerance, "composite stencil scaling applied once");
}
std::vector<Real> ch(num::Backend backend, num::Fusion fusion) {
    auto m = mesh();
    ScalarField c(m, "c", 1), mu(m, "mu", 1);
    init(c);
    if (backend == num::Backend::CUDA) {
        device(c);
        device(mu);
    }
    auto sch = Schemes::builtin();
    num::Spatial ops(sch);
    num::System sys(sch);
    PeriodicBC px(m.patch("xmin")), py(m.patch("ymin"));
    sys.bc(c, {&px, &py});
    sys.bc(mu, {&px, &py});
    sys.add(num::ddt(c) == .05 * ops.lap(mu)); // deliberately register consumer first
    sys.define(mu, ops.pw(c, pure(Nonlinear{})) - .002 * ops.lap(c));
    sys.compile({backend, fusion});
    const auto initial = mass(c);
    for (int i = 0; i < 15; ++i)
        sys.advance(1e-4);
    sys.refresh(mu);
    if (backend == num::Backend::CUDA) {
        c.downloadCurrFromDevice();
        mu.downloadCurrFromDevice();
    }
    require(std::abs(mass(c) - initial) < tolerance, "CH mass conservation");
    require(std::abs(sys.time() - .0015) < 1e-15 && sys.step() == 15, "one macrostep clock");
    ScalarField current(m, "current", 1);
    periodic(c);
    evaluate(num::collapse(PhiX::pw(c, Nonlinear{}) - .002 * PhiX::lap(c)), current, false);
    require(diff(current, mu) < tolerance, "auxiliary refreshed at current output time");
    sys.restoreClock(.1, 100);
    sys.advance(1e-4);
    require(sys.step() == 101 && std::abs(sys.time() - .1001) < 1e-15, "restart clock");
    std::ostringstream report;
    sys.report(report);
    require(report.str().find("ForwardEuler") != std::string::npos, "execution report");
    if (backend == num::Backend::CUDA)
        c.downloadCurrFromDevice();
    return c.curr;
}
void systems() {
    auto cpu = ch(num::Backend::CPU, num::Fusion::Off);
    auto off = ch(num::Backend::CUDA, num::Fusion::Off);
    auto on = ch(num::Backend::CUDA, num::Fusion::Auto);
    double err = 0;
    for (size_t i = 0; i < cpu.size(); ++i)
        err = std::max(err, std::abs(double(cpu[i]) - double(off[i])) +
                                std::abs(double(on[i]) - double(off[i])));
    require(err < tolerance, "CH CPU/ordinary/fused trajectory");
    auto m = mesh();
    auto sch = Schemes::builtin();
    num::Spatial ops(sch);
    for (auto coupling : {num::Coupling::SameLevel, num::Coupling::Sequential}) {
        ScalarField a(m, "a"), b(m, "b");
        a.fill(1);
        b.fill(2);
        num::System s(sch);
        s.add(num::ddt(a) == ops.value(b));
        s.add(num::ddt(b) == ops.value(a));
        s.compile({num::Backend::CPU, num::Fusion::Auto, coupling});
        s.advance(.1);
        require(std::abs(a.curr[a.index(0, 0)] - 1.2) < tolerance, "Euler a");
        require(std::abs(b.curr[b.index(0, 0)] -
                         (coupling == num::Coupling::SameLevel ? 2.1 : 2.12)) < tolerance,
                "coupling semantics retained");
    }
    ScalarField a(m, "a"), b(m, "b");
    {
        num::System s(sch);
        s.define(a, ops.value(b));
        s.define(b, ops.value(a));
        rejects([&] { s.compile({num::Backend::CPU}); }, "auxiliary cycle must fail");
    }
    {
        num::System s(sch);
        s.define(a, ops.value(b));
        s.add(num::ddt(a) == ops.value(b));
        rejects([&] { s.compile({num::Backend::CPU}); }, "duplicate writer must fail");
    }
    {
        num::System s(sch);
        s.add(num::ddt(a) == ops.lap(a));
        rejects([&] { s.compile({num::Backend::CPU}); }, "missing BC must fail at setup");
    }
    {
        auto rk = Schemes::builtin({{"ddt", "RK4"}});
        num::System s(rk);
        s.add(num::ddt(a) == ops.value(a));
        rejects([&] { s.compile({num::Backend::CPU}); }, "unsupported time scheme must fail");
    }
    {
        Term unknown;
        unknown.field = &a;
        unknown.cpu_kernel = [](Real *, double, ScratchPool &) {};
        num::System s(sch);
        s.define(b, unknown);
        rejects([&] { s.compile({num::Backend::CPU}); }, "unknown dependency contract must fail");
    }
    {
        a.fill(1);
        b.fill(0);
        num::System s(sch);
        s.define(b, ops.value(a) * 2);
        s.add(num::ddt(a) == ops.value(b));
        num::ExternalStep projection;
        projection.name = "clamp";
        projection.reads.complete = true;
        projection.reads.read(&a);
        projection.writes = {&a};
        projection.cpu = [&](double, double) { a.fill(3); };
        s.afterStep(projection);
        s.compile({num::Backend::CPU});
        s.advance(.1);
        s.refresh(b);
        require(std::abs(b.curr[b.index(0, 0)] - 6) < tolerance,
                "external write invalidates auxiliary");
        std::ostringstream report;
        s.report(report);
        require(report.str().find("afterStep clamp: external barrier") != std::string::npos,
                "model barriers visible in execution report");
    }
    {
        a.fill(1);
        num::System s(sch);
        s.add(num::ddt(a) == ops.value(a));
        s.compile({num::Backend::CPU});
        s.advanceAdaptive(
            [](const std::vector<const ScalarField *> &rhs) { return rhs.size() == 1 ? .1 : 0; });
        require(std::abs(a.curr[a.index(0, 0)] - 1.1) < tolerance, "adaptive RHS barrier");
    }
    {
        num::System s(sch);
        s.add(num::ddt(a) == ops.value(a));
        s.compile({num::Backend::CPU});
        ++a.storedDims[0];
        rejects([&] { s.advance(.1); }, "layout mutation rejected before execution");
        --a.storedDims[0];
    }
    {
        device(a);
        auto shell = ScalarField::makeShell(a.mesh, a.ghost, a.d_curr);
        num::System s(sch);
        s.add(num::ddt(a) == ops.value(shell));
        rejects([&] { s.compile({num::Backend::CUDA}); },
                "distinct aliased graph handles rejected");
    }
    {
        num::System s(sch);
        num::ExternalStep bad;
        bad.reads.complete = true;
        bad.writes = {nullptr};
        bad.cpu = [](double, double) {};
        s.afterStep(bad);
        rejects([&] { s.compile({num::Backend::CPU}); }, "null external output rejected");
    }
    {
        ScalarField wide(m, "wide", 2);
        auto high = Schemes::builtin({{"laplacian", "CD4"}});
        num::Spatial highops(high);
        num::System s(high);
        NoFluxBC xl(m.patch("xmin")), xh(m.patch("xmax")), yl(m.patch("ymin")), yh(m.patch("ymax"));
        s.bc(wide, {&xl, &xh, &yl, &yh});
        s.add(num::ddt(wide) == highops.lap(wide));
        rejects([&] { s.compile({num::Backend::CPU}); }, "wide NoFlux rejected");
    }
}
void versionsAndMulti() {
    auto m = mesh();
    auto sch = Schemes::builtin();
    num::Spatial ops(sch);
    ScalarField a(m, "a"), b(m, "b"), x(m, "x"), y(m, "y"), z(m, "z");
    a.fill(2);
    b.fill(3);
    for (auto *f : {&a, &b, &x, &y, &z})
        device(*f);
    for (auto mode : {num::Fusion::Off, num::Fusion::Auto, num::Fusion::Required}) {
        num::System sys(sch);
        num::defineMany(sys, x, ops.value(a) * 2, y, ops.value(a) * ops.value(b), z,
                        ops.pw(b, pure(Square{})));
        sys.compile({num::Backend::CUDA, mode});
        sys.refresh(y);
        x.downloadCurrFromDevice();
        y.downloadCurrFromDevice();
        z.downloadCurrFromDevice();
        require(std::abs(x.curr[x.index(0, 0)] - 4) < tolerance &&
                    std::abs(y.curr[y.index(0, 0)] - 6) < tolerance &&
                    std::abs(z.curr[z.index(0, 0)] - 9) < tolerance,
                "multi-output ordinary/fused result");
    }
    Resource table{"material"};
    int evaluations = 0;
    num::System sys(sch);
    num::ExternalStep producer;
    producer.name = "material lookup";
    producer.reads.complete = true;
    producer.reads.read(&a);
    producer.reads.read(&table);
    producer.writes = {&x};
    producer.cpu = [&](double, double) {
        ++evaluations;
        x.fill(Real(table.version + 1));
    };
    sys.producer(producer);
    sys.define(y, ops.value(x) * 2);
    sys.compile({num::Backend::CPU});
    sys.refresh(y);
    sys.refresh(y);
    require(evaluations == 1, "auxiliary cache reuses unchanged dependencies");
    table.touch();
    sys.refresh(y);
    require(evaluations == 2 && std::abs(y.curr[y.index(0, 0)] - 4) < tolerance,
            "external resource version invalidation");
    sys.replace(y, ops.value(x) * 3);
    sys.refresh(y);
    require(std::abs(y.curr[y.index(0, 0)] - 6) < tolerance, "replace coefficient propagates");
    rejects([&] { sys.replace(y, ops.value(b)); }, "replace cannot change graph dependencies");
}
void faces() {
    auto m = mesh();
    ScalarField c(m, "c"), mob(m, "mob"), cpu(m, "cpu"), off(m, "off"), on(m, "on");
    init(c);
    mob.initialize([](double x, double y, double) { return 1 + x * x + 2 * y; });
    periodic(c);
    periodic(mob);
    device(c);
    device(mob);
    device(off);
    device(on);
    auto sch = Schemes::builtin();
    num::Faces ops(sch);
    auto flux = ops.flux("diffusion",
                         [&](auto face) { return ops.interp(mob, face) * ops.snGrad(c, face); });
    auto expression = ops.div(flux);
    evaluate(expression.lower(num::Fusion::Off), cpu, false);
    evaluate(expression.lower(num::Fusion::Off), off, true);
    evaluate(expression.lower(num::Fusion::Required), on, true);
    require(diff(cpu, off) < tolerance, "materialized face CPU/CUDA");
    require(diff(off, on) < tolerance, "certified face assembled equality including boundaries");
    require(std::abs(mass(on)) < tolerance * 20, "variable periodic mobility conserves mass");
    {
        ScratchPool pool;
        CUDA_CHECK(cudaStreamCreateWithFlags(&pool.stream, cudaStreamNonBlocking));
        Real *guarded = nullptr;
        CUDA_CHECK(cudaMalloc(&guarded, (on.storedSize + 2) * sizeof(Real)));
        const Real guard = Real(12345);
        std::vector<Real> host(on.storedSize + 2, guard);
        for (auto mode : {num::Fusion::Off, num::Fusion::Required}) {
            CUDA_CHECK(cudaMemcpyAsync(guarded, host.data(), host.size() * sizeof(Real),
                                       cudaMemcpyHostToDevice, pool.stream));
            CUDA_CHECK(cudaMemsetAsync(guarded + 1, 0, on.storedBytes(), pool.stream));
            pool.reset();
            auto t = expression.lower(mode);
            t.gpu_launcher(guarded + 1, t.coeff, pool);
            CUDA_CHECK(cudaStreamSynchronize(pool.stream));
            CUDA_CHECK(cudaMemcpy(host.data(), guarded, host.size() * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
            require(host.front() == guard && host.back() == guard,
                    "face divergence preserves output buffer guards");
            double error = 0;
            for (int j = 0; j < m.n[1]; ++j)
                for (int i = 0; i < m.n[0]; ++i) {
                    const auto k = on.index(i, j);
                    error = std::max(error, std::abs(double(host[k + 1] - off.curr[k])));
                }
            require(error < tolerance, "face graph honors explicit nonblocking stream");
        }
        pool.reset();
        auto *first = num::face_detail::acquire<0>(c, num::Backend::CUDA, pool);
        pool.reset();
        auto rebuilt = expression.lower(num::Fusion::Off);
        CUDA_CHECK(cudaMemsetAsync(guarded + 1, 0, on.storedBytes(), pool.stream));
        rebuilt.gpu_launcher(guarded + 1, 1, pool);
        CUDA_CHECK(cudaStreamSynchronize(pool.stream));
        pool.reset();
        require(first == num::face_detail::acquire<0>(c, num::Backend::CUDA, pool),
                "rebuilt face expression reuses the equation scratch pool");
        CUDA_CHECK(cudaFree(guarded));
        CUDA_CHECK(cudaStreamDestroy(pool.stream));
        pool.stream = nullptr;
    }
    FaceField fx(m, 0, "fx"), fy(m, 1, "fy");
    fx.allocDevice();
    fy.allocDevice();
    num::System system(sch);
    PeriodicBC px(m.patch("xmin")), py(m.patch("ymin"));
    system.bc(c, {&px, &py});
    system.bc(mob, {&px, &py});
    system.define(on, num::adapted(PhiX::divFace(fx, fy)));
    num::defineFace(system, fx, flux.x);
    num::defineFace(system, fy, flux.y);
    system.compile({num::Backend::CUDA});
    system.refresh(on);
    on.downloadCurrFromDevice();
    require(diff(off, on) < tolerance, "face producers scheduled before div consumer");
    rejects([&] { flux.x.evaluate(fy, num::Backend::CPU, num::Fusion::Off); },
            "face axis mismatch rejected");
    mob.fill(2);
    mob.uploadAllToDevice();
    system.touch(mob);
    system.refresh(on);
    on.downloadCurrFromDevice();
    evaluate(2 * PhiX::lap(c), cpu, false);
    require(diff(cpu, on) < tolerance, "face chain version refresh");
    // Face gradient vector reconstruction matches existing assembled primitive.
    auto vectorFlux = ops.flux(
        "grad", [&](auto face) { return ops.gradComponent(c, face, decltype(face)::value); });
    evaluate(ops.div(vectorFlux).lower(num::Fusion::Auto), on, true);
    evaluate(PhiX::lap(c), cpu, false);
    require(diff(cpu, on) < tolerance, "face gradient normal primitive");
}
} // namespace
int main() {
    try {
        auto start = std::chrono::steady_clock::now();
        config();
        fragments();
        systems();
        versionsAndMulti();
        faces();
        std::cout << "numerics: " << checks << " checks passed; "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " s\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAILED after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
