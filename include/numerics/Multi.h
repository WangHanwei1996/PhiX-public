#pragma once
#include "numerics/System.h"
#include <set>

namespace PhiX::numerics {
namespace multi_detail {
template <class S, class A, class B, class C>
void launch(const ScalarField &layout, ScalarField &x, const A &a, ScalarField &y, const B &b,
            ScalarField &z, const C &c) {
    Fused::fuse_multi_compute(layout, x, detail::Compile<decltype(a.node), S>::get(a.node), y,
                              detail::Compile<decltype(b.node), S>::get(b.node), z,
                              detail::Compile<decltype(c.node), S>::get(c.node));
}
} // namespace multi_detail
// A declared pure three-output region. No automatic cross-definition rewriting.
template <class A, bool FA, class B, bool FB, class C, bool FC>
void defineMany(System &system, ScalarField &x, CellExpr<A, FA> a, ScalarField &y,
                CellExpr<B, FB> b, ScalarField &z, CellExpr<C, FC> c) {
    check::checkSameMesh(x, y, "defineMany");
    check::checkSameMesh(x, z, "defineMany");
    auto qx = std::make_shared<Equation>(x), qy = std::make_shared<Equation>(y),
         qz = std::make_shared<Equation>(z);
    qx->setRHS(a.reference);
    qy->setRHS(b.reference);
    qz->setRHS(c.reference);
    struct State {
        bool fused = false;
        std::string scheme = "CD2";
        std::string report;
    };
    auto state = std::make_shared<State>();
    ExternalStep step;
    step.name = "multi-output(" + x.name + "," + y.name + "," + z.name + ")";
    step.reads = operationInfo(a.reference);
    step.reads.merge(operationInfo(b.reference));
    step.reads.merge(operationInfo(c.reference));
    for (const auto &r : step.reads.reads)
        if (r.cell)
            check::checkSameMesh(*r.cell, x, "defineMany input/output layout");
    step.writes = {&x, &y, &z};
    step.configure = [=](Backend backend, Fusion mode) {
        std::set<std::string> schemes;
        if (a.stencils)
            schemes.insert(a.stencilScheme);
        if (b.stencils)
            schemes.insert(b.stencilScheme);
        if (c.stencils)
            schemes.insert(c.stencilScheme);
        const bool supported = FA && FB && FC && a.stencils <= 1 && b.stencils <= 1 &&
                               c.stencils <= 1 && a.reason.empty() && b.reason.empty() &&
                               c.reason.empty() && schemes.size() <= 1;
        if (mode == Fusion::Required && !supported)
            throw std::invalid_argument(
                "defineMany: requested multi-output fusion cannot be generated");
        state->fused = mode != Fusion::Off && supported;
        if (!schemes.empty())
            state->scheme = *schemes.begin();
        if (backend == Backend::CPU) {
            state->report = "CPU reference for declared three-output region";
            return;
        }
        state->report = state->fused
                            ? "fused three-output region"
                            : "three materialized expressions (fusion off or unsupported region)";
    };
    step.describe = [=] { return state->report; };
    step.cpu = [=, &x, &y, &z](double, double) {
        qx->computeRHSCPU(x);
        qy->computeRHSCPU(y);
        qz->computeRHSCPU(z);
    };
    step.gpu = [=, &x, &y, &z](double, double) {
        if constexpr (FA && FB && FC) {
            if (state->fused) {
                if (state->scheme == "CD2")
                    multi_detail::launch<scheme::CD2>(x, x, a, y, b, z, c);
                else if (state->scheme == "CD4")
                    multi_detail::launch<scheme::CD4>(x, x, a, y, b, z, c);
                else if (state->scheme == "CD6")
                    multi_detail::launch<scheme::CD6>(x, x, a, y, b, z, c);
                else if (state->scheme == "Iso9")
                    multi_detail::launch<scheme::Iso9>(x, x, a, y, b, z, c);
                else
                    throw std::logic_error("defineMany: unbound scheme");
                return;
            }
        }
        qx->computeRHS(x);
        qy->computeRHS(y);
        qz->computeRHS(z);
    };
    system.producer(std::move(step));
}
} // namespace PhiX::numerics
