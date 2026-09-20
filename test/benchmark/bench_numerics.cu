#include "numerics/Face.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#ifdef PHIX_HAVE_CUPTI
#include <cupti.h>
#endif
using namespace PhiX;
namespace num = PhiX::numerics;
struct Nonlinear {
    __host__ __device__ Real operator()(Real x) const { return x * x * x - x; }
};
struct Measurements {
    unsigned long long launches = 0;
    std::size_t liveBytes = 0, peakBytes = 0;
    std::map<void *, std::size_t> allocations;
    bool countLaunches = false;
};
#ifdef PHIX_HAVE_CUPTI
static void CUPTIAPI callback(void *user, CUpti_CallbackDomain domain, CUpti_CallbackId,
                              const void *data) {
    auto &stats = *static_cast<Measurements *>(user);
    auto *info = static_cast<const CUpti_CallbackData *>(data);
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API)
        return;
    if (info->callbackSite == CUPTI_API_ENTER && stats.countLaunches &&
        std::strncmp(info->functionName, "cudaLaunchKernel", 16) == 0)
        ++stats.launches;
    if (info->callbackSite != CUPTI_API_EXIT || !info->functionReturnValue ||
        *static_cast<const cudaError_t *>(info->functionReturnValue) != cudaSuccess)
        return;
    if (std::strcmp(info->functionName, "cudaMalloc") == 0) {
        auto *p = static_cast<const cudaMalloc_v3020_params *>(info->functionParams);
        stats.allocations[*p->devPtr] = p->size;
        stats.liveBytes += p->size;
        stats.peakBytes = std::max(stats.peakBytes, stats.liveBytes);
    } else if (std::strcmp(info->functionName, "cudaFree") == 0) {
        auto *p = static_cast<const cudaFree_v3020_params *>(info->functionParams);
        auto it = stats.allocations.find(p->devPtr);
        if (it != stats.allocations.end()) {
            stats.liveBytes -= it->second;
            stats.allocations.erase(it);
        }
    }
}
#endif
template <class Expr>
void measure(const char *name, const Expr &expr, ScalarField &output, num::Fusion mode) {
    size_t freeBefore, total, freeAfter;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemGetInfo(&freeBefore, &total));
    Measurements stats;
    bool counted = false;
#ifdef PHIX_HAVE_CUPTI
    CUpti_SubscriberHandle subscriber;
    bool subscribed = cuptiSubscribe(&subscriber, callback, &stats) == CUPTI_SUCCESS;
    if (subscribed)
        counted = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API) == CUPTI_SUCCESS;
#endif
    Equation equation(output);
    equation.setRHS(expr.lower(mode));
    for (int i = 0; i < 10; ++i)
        equation.computeRHS(output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemGetInfo(&freeAfter, &total));
    stats.countLaunches = true;
    equation.computeRHS(output);
    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef PHIX_HAVE_CUPTI
    if (subscribed)
        cuptiUnsubscribe(subscriber);
#endif
    const int iterations = 200;
    std::array<double, 5> samples;
    for (auto &sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            equation.computeRHS(output);
        CUDA_CHECK(cudaDeviceSynchronize());
        sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count() /
                 iterations;
    }
    std::sort(samples.begin(), samples.end());
    const double ms = samples[samples.size() / 2];
    std::cout << name << "," << (mode == num::Fusion::Off ? "off" : "auto") << ","
              << output.mesh.n[0] << "," << ms << ","
              << (freeBefore > freeAfter ? freeBefore - freeAfter : 0) << ",";
    if (counted)
        std::cout << stats.peakBytes << "," << stats.launches;
    else
        std::cout << "unavailable,unavailable";
    std::cout << "\n";
}
int main() {
    try {
        std::cout << "expression,fusion,n,median_wall_ms_per_eval,resident_extra_device_bytes,peak_"
                     "expression_allocated_bytes,measured_"
                     "kernel_launches\n";
        for (int n : {128, 512}) {
            Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, n, 1.0 / n, 0, n, 1.0 / n, 0);
            ScalarField c(mesh, "c"), mob(mesh, "mob"), out(mesh, "out");
            c.fill(.2);
            mob.fill(.05);
            for (auto *f : {&c, &mob, &out}) {
                f->allocDevice();
                f->uploadAllToDevice();
            }
            auto sch = Schemes::builtin();
            num::Spatial ops(sch);
            num::Faces faces(sch);
            auto cell = ops.pw(c, num::pure(Nonlinear{})) - .002 * ops.lap(c);
            auto flux = faces.div(faces.flux(
                "J", [&](auto axis) { return faces.interp(mob, axis) * faces.snGrad(c, axis); }));
            for (auto mode : {num::Fusion::Off, num::Fusion::Auto})
                measure("cell", cell, out, mode);
            for (auto mode : {num::Fusion::Off, num::Fusion::Auto})
                measure("flux", flux, out, mode);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
