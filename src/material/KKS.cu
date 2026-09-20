#include "material/KKS.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

// ===========================================================================
// KKSParabolic
// ===========================================================================

KKSParabolic::KKSParabolic(double ks, double cs0, double kl, double cl0,
                           double bs, double bl)
    : ks_(ks), cs0_(cs0), bs_(bs), kl_(kl), cl0_(cl0), bl_(bl)
{
    if (ks_ <= 0.0 || kl_ <= 0.0)
        throw std::invalid_argument(
            "KKSParabolic: curvatures ks, kl must be > 0");
}

KKSView KKSParabolic::view() const {
    KKSView v;
    v.ks  = static_cast<Real>(ks_);
    v.cs0 = static_cast<Real>(cs0_);
    v.bs  = static_cast<Real>(bs_);
    v.kl  = static_cast<Real>(kl_);
    v.cl0 = static_cast<Real>(cl0_);
    v.bl  = static_cast<Real>(bl_);
    return v;
}

KKSParabolic::Equilibrium KKSParabolic::equilibrium() const {
    // μ²·A + μ·B + C = 0  with
    //   A = 1/(2 kl) − 1/(2 ks),  B = cl0 − cs0,  C = bs − bl
    const double A = 0.5 / kl_ - 0.5 / ks_;
    const double B = cl0_ - cs0_;
    const double C = bs_ - bl_;

    double mu;
    if (std::fabs(A) < 1e-300) {
        if (std::fabs(B) < 1e-300)
            throw std::runtime_error(
                "KKSParabolic::equilibrium: degenerate (ks==kl and cs0==cl0)");
        mu = -C / B;
    } else {
        const double disc = B * B - 4.0 * A * C;
        if (disc < 0.0)
            throw std::runtime_error(
                "KKSParabolic::equilibrium: no real common-tangent solution");
        const double sq = std::sqrt(disc);
        const double r1 = (-B + sq) / (2.0 * A);
        const double r2 = (-B - sq) / (2.0 * A);
        mu = (std::fabs(r1) <= std::fabs(r2)) ? r1 : r2;
    }

    return {mu, cs0_ + mu / ks_, cl0_ + mu / kl_};
}

// ===========================================================================
// Field-level partition
// ===========================================================================

namespace {

__global__ void kernel_kks_partition(
        KKSView v,
        const Real* c, const Real* h,
        Real* cs, Real* cl, Real* mu,
        std::size_t n)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Real csv, clv, muv;
    v.partition(c[tid], h[tid], csv, clv, muv);
    cs[tid] = csv;
    cl[tid] = clv;
    mu[tid] = muv;
}

void checkLayouts(const ScalarField& c, const ScalarField& h,
                  const ScalarField& cs, const ScalarField& cl,
                  const ScalarField& mu, const char* fn)
{
    const ScalarField* fs[4] = {&h, &cs, &cl, &mu};
    for (const ScalarField* f : fs)
        if (f->storedSize != c.storedSize || f->ghost != c.ghost)
            throw std::invalid_argument(
                std::string(fn) + ": field '" + f->name
                + "' layout differs from '" + c.name + "'");
}

} // namespace

void kksPartitionOnGPU(const KKSParabolic& model,
                       const ScalarField& c, const ScalarField& hFrac,
                       ScalarField& cs, ScalarField& cl, ScalarField& mu)
{
    checkLayouts(c, hFrac, cs, cl, mu, "kksPartitionOnGPU");
    if (!c.d_curr || !hFrac.d_curr || !cs.d_curr || !cl.d_curr || !mu.d_curr)
        throw std::runtime_error(
            "kksPartitionOnGPU: all fields need device allocation");

    const std::size_t n = c.storedSize;
    const int blocks = static_cast<int>((n + 255) / 256);
    kernel_kks_partition<<<blocks, 256>>>(
        model.view(), c.d_curr, hFrac.d_curr,
        cs.d_curr, cl.d_curr, mu.d_curr, n);
    CUDA_CHECK(cudaGetLastError());
}

void kksPartitionOnCPU(const KKSParabolic& model,
                       const ScalarField& c, const ScalarField& hFrac,
                       ScalarField& cs, ScalarField& cl, ScalarField& mu)
{
    checkLayouts(c, hFrac, cs, cl, mu, "kksPartitionOnCPU");
    const KKSView v = model.view();
    const std::size_t n = c.storedSize;
    for (std::size_t i = 0; i < n; ++i) {
        Real csv, clv, muv;
        v.partition(c.curr[i], hFrac.curr[i], csv, clv, muv);
        cs.curr[i] = csv;
        cl.curr[i] = clv;
        mu.curr[i] = muv;
    }
}

} // namespace PhiX
