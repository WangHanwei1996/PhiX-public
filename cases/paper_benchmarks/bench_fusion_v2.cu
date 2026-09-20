// ---------------------------------------------------------------------------
// bench_fusion_v2 — Off vs Auto on the symbolic (original-equation) path.
//
// Replaces bench_fusion.cu (EvalPlan vs hand-written FusedTerm) for the
// v3.10+ architecture: the user writes the continuous expression, the
// framework expands it, binds schemes and lowers it to one CUDA kernel
// (Fusion::Auto) or materialises every unique derivative primitive first
// (Fusion::Off).  Same protocol as the old benchmark: 5 warm-up + 50 timed
// evaluations per repeat, 4 repeats, ranges reported; max|Off-Auto| on the
// physical cells.
//
// Cases
//   mu10   the ten-term MPF_AC_DW chemical potential mu_0 (2 pointwise
//          double-well + 8 gradient/Laplacian cross terms), symbolic form
//   aniso  the expanded anisotropic divergence of the TK2015 phase equation
//          (bench_symbolic's expression), symbolic form
//
// Kernel launches are (a) the plan's own count and (b) measured in-process by
// CUPTI on the driver API (cuLaunchKernel*), which sees both runtime-launched
// and NVRTC/driver-launched kernels.  Timing runs with callbacks detached.
// ---------------------------------------------------------------------------
#include "numerics/Symbolic.h"
#include "boundary/PeriodicBC.h"
#include "core/CudaCheck.h"
#include "core/Version.h"
#include <cupti.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <functional>
#include <string>
#include <vector>

using namespace PhiX;
namespace num = PhiX::numerics;
namespace sym = num::symbolic;

struct LaunchCount {
    unsigned long long driver = 0, runtime = 0;
    bool on = false;
};
static void CUPTIAPI onApi(void* user, CUpti_CallbackDomain domain, CUpti_CallbackId,
                           const void* data) {
    auto& c = *static_cast<LaunchCount*>(user);
    auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (!c.on || info->callbackSite != CUPTI_API_ENTER) return;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
        std::strncmp(info->functionName, "cuLaunchKernel", 14) == 0) ++c.driver;
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        std::strncmp(info->functionName, "cudaLaunchKernel", 16) == 0) ++c.runtime;
}

static double timeIt(const std::function<void()>& fn, int warm, int iters) {
    for (int i = 0; i < warm; ++i) fn();
    PHIX_CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    PHIX_CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

struct Result {
    std::string name;
    int N = 0;
    std::size_t planOff = 0, planAuto = 0;
    unsigned long long cuptiOff = 0, cuptiAuto = 0, rtOff = 0, rtAuto = 0;
    std::vector<double> msOff, msAuto;
    double maxdiff = 0;
};

// Run one expression through Equation::computeRHS in both modes.
static Result run(const std::string& name, int N, const sym::Expanded& expanded,
                  ScalarField& out, std::vector<ScalarField*> inputs, int repeats) {
    auto plan = expanded.discretize(Schemes::builtin());
    Result r;
    r.name = name;
    r.N = N;
    r.planOff = plan.kernelLaunches(num::Fusion::Off);
    r.planAuto = plan.kernelLaunches(num::Fusion::Auto);

    std::vector<Real> refOff;
    for (auto mode : {num::Fusion::Off, num::Fusion::Auto}) {
        Equation eq(out);
        eq.setRHS(plan.lower(mode));
        // first evaluation: scratch allocation / NVRTC compile, excluded from timing
        eq.computeRHS(out);
        PHIX_CUDA_CHECK(cudaDeviceSynchronize());

        // launch count: exactly one evaluation under CUPTI, callbacks then detached
        LaunchCount lc;
        CUpti_SubscriberHandle sub;
        bool subscribed = cuptiSubscribe(&sub, onApi, &lc) == CUPTI_SUCCESS;
        if (subscribed) {
            cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API);
            cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API);
            lc.on = true;
            eq.computeRHS(out);
            PHIX_CUDA_CHECK(cudaDeviceSynchronize());
            lc.on = false;
            cuptiUnsubscribe(sub);
        }
        std::vector<double> ms;
        for (int k = 0; k < repeats; ++k)
            ms.push_back(timeIt([&] { eq.computeRHS(out); }, 5, 50));
        out.downloadCurrFromDevice();
        if (mode == num::Fusion::Off) {
            r.msOff = ms; r.cuptiOff = lc.driver; r.rtOff = lc.runtime;
            refOff = out.curr;
        } else {
            r.msAuto = ms; r.cuptiAuto = lc.driver; r.rtAuto = lc.runtime;
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i) {
                    auto c = out.index(i, j, 0);
                    if (!std::isfinite(out.curr[c])) throw std::runtime_error("non-finite result");
                    r.maxdiff = std::max(r.maxdiff, std::abs(double(out.curr[c] - refOff[c])));
                }
        }
    }
    (void)inputs;
    return r;
}

static void print(const Result& r) {
    auto mm = [](const std::vector<double>& v) {
        return std::make_pair(*std::min_element(v.begin(), v.end()),
                              *std::max_element(v.begin(), v.end()));
    };
    auto o = mm(r.msOff), a = mm(r.msAuto);
    const double cells = double(r.N) * r.N;
    std::printf("%-6s N=%5d  plan kernels Off=%zu Auto=%zu | CUPTI cuLaunchKernel Off=%llu Auto=%llu"
                " (cudaLaunchKernel Off=%llu Auto=%llu)\n",
                r.name.c_str(), r.N, r.planOff, r.planAuto, r.cuptiOff, r.cuptiAuto, r.rtOff,
                r.rtAuto);
    std::printf("        Off  %8.3f--%8.3f ms/eval  %9.1f--%9.1f Mcells/s\n", o.first, o.second,
                cells / (o.second * 1e3), cells / (o.first * 1e3));
    std::printf("        Auto %8.3f--%8.3f ms/eval  %9.1f--%9.1f Mcells/s   speedup %4.2f--%4.2fx\n",
                a.first, a.second, cells / (a.second * 1e3), cells / (a.first * 1e3),
                o.first / a.second, o.second / a.first);
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return 0.5 * (v[1] + v[2]); };
    std::printf("        median Off %8.4f  Auto %8.4f ms/eval  (median of 4: mean of middle two)\n", med(r.msOff), med(r.msAuto));
    std::printf("        max|Off-Auto| = %.3e\n\n", r.maxdiff);
}

int main() {
    try {
        std::printf("bench_fusion_v2  %s  Real=%zu bytes  repeats=4 x (5 warm + 50 timed)\n\n",
                    PhiX::versionString(), sizeof(Real));
        const double W = 1.0, e2a = 0.4, e2b = 0.4;
        const int repeats = 4;

        for (int N : {512, 1024, 2048}) {
            const double dx = 1.0 / N;
            Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

            // ---- case mu10: same fields, same initial data as bench_fusion.cu ----
            ScalarField phi0(mesh, "phi0", 1), phia(mesh, "phia", 1), phib(mesh, "phib", 1);
            phi0.initialize([](double x, double y, double) {
                return 0.5 + 0.4 * std::sin(6.28318 * x) * std::cos(6.28318 * y); });
            phia.initialize([](double x, double y, double) {
                return 0.25 + 0.2 * std::cos(12.566 * x) * std::sin(6.28318 * y); });
            phib.initialize([](double x, double y, double) {
                return 0.25 + 0.2 * std::sin(12.566 * y); });
            PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
            PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
            for (ScalarField* f : {&phi0, &phia, &phib}) {
                f->allocDevice(); f->uploadAllToDevice();
                bx.applyOnGPU(*f); by.applyOnGPU(*f);
            }
            ScalarField mu(mesh, "mu", 1);
            mu.allocDevice();

            // mu_0 of MPF_AC_DW, written as the continuous expression
            const auto P = sym::field(phi0), A = sym::field(phia), B = sym::field(phib);
            const auto gP = sym::grad(P), gA = sym::grad(A), gB = sym::grad(B);
            sym::Expr mu0 =
                P * A * A * (2.0 * W) +
                P * B * B * (2.0 * W) +
                P * sym::dot(gA, gA) * (2.0 * e2a) -
                A * sym::dot(gP, gA) * (2.0 * e2a) +
                P * A * sym::lap(A) * e2a -
                A * A * sym::lap(P) * e2a +
                P * sym::dot(gB, gB) * (2.0 * e2b) -
                B * sym::dot(gP, gB) * (2.0 * e2b) +
                P * B * sym::lap(B) * e2b -
                B * B * sym::lap(P) * e2b;
            auto ex10 = mu0.expand("mu10");
            if (N == 512) { ex10.print(std::cout); std::printf("\n"); }
            print(run("mu10", N, ex10, mu, {&phi0, &phia, &phib}, repeats));

            // ---- case aniso: bench_symbolic's expanded anisotropic divergence ----
            ScalarField psi(mesh, "psi", 1), theta(mesh, "theta", 1), out(mesh, "out", 1);
            psi.initialize([](double x, double y, double) {
                return 0.2 * x + 0.1 * y + std::sin(0.1 * x) * std::cos(0.12 * y); });
            theta.fill(0.25);
            for (ScalarField* f : {&psi, &theta}) {
                f->allocDevice(); f->uploadAllToDevice();
                bx.applyOnGPU(*f); by.applyOnGPU(*f);
            }
            out.allocDevice();
            const auto s = sym::field(psi), orientation = sym::parameter(theta);
            const auto g = sym::grad(s);
            const auto angle = sym::atan2(g.y, g.x);
            const auto a = 1 + 0.007 * sym::cos(4 * (angle - orientation));
            auto exA = sym::div(a * a * g + a * sym::partial(a, angle) * sym::perpendicular(g))
                           .expand("aniso");
            if (N == 512) { exA.print(std::cout); std::printf("\n"); }
            print(run("aniso", N, exA, out, {&psi, &theta}, repeats));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
