#include "mechanics/ElasticityFFT.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// cuFFT error checking — local to this TU (the central core/CudaCheck.h
// stays cuFFT-free so its includers don't pull in <cufft.h>).
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult _r = (call);                                               \
        if (_r != CUFFT_SUCCESS)                                               \
            throw ::PhiX::DeviceError(                                         \
                std::string(__FILE__) + ":" + std::to_string(__LINE__),        \
                "cuFFT error code "                                            \
                + std::to_string(static_cast<int>(_r)) + " in " #call);        \
    } while (0)

// Real-type-dependent cuFFT bindings
#ifdef PHIX_REAL_FLOAT
using CufftComplexT = cufftComplex;
static constexpr cufftType FFT_R2C = CUFFT_R2C;
static constexpr cufftType FFT_C2R = CUFFT_C2R;
#else
using CufftComplexT = cufftDoubleComplex;
static constexpr cufftType FFT_R2C = CUFFT_D2Z;
static constexpr cufftType FFT_C2R = CUFFT_Z2D;
#endif

void ElasticParams2D::validate() const {
    if (C44 <= 0.0 || C11 <= 0.0)
        throw std::invalid_argument("ElasticParams2D: C11, C44 must be > 0");
    if (C11 - C12 <= 0.0 || C11 + C12 <= 0.0)
        throw std::invalid_argument(
            "ElasticParams2D: need C11 > |C12| (positive definiteness)");
}

namespace {

// pack/unpack between ghost-padded ScalarFields and dense nx·ny arrays
__global__ void kernel_pack(Real* out, const Real* f,
                            int nx, int ny, int sx, int sy, int g)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx;
    const int j = idx / nx;
    out[idx] = f[(i + g) + sx * ((j + g) + sy * g)];
}

__global__ void kernel_unpack(Real* f, const Real* in, Real scale,
                              int nx, int ny, int sx, int sy, int g)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx;
    const int j = idx / nx;
    f[(i + g) + sx * ((j + g) + sy * g)] = in[idx] * scale;
}

// ---------------------------------------------------------------------------
// Per-mode Khachaturyan solution.  For dilatational eigenstrain the
// eigenstress is σ*_ij = s·δ_ij with s = (C11 + C12)·ê*, so v_k = σ*_kl ξ_l
// = s·ξ_k and with w = K⁻¹·ξ (acoustic tensor K):
//     ε̂11 = s·w1·ξ1,   ε̂22 = s·w2·ξ2,   ε̂12 = ½·s·(w1ξ2 + w2ξ1)
// The zero mode carries the homogeneous strain (set on the host side).
// ---------------------------------------------------------------------------
__global__ void kernel_modes(const CufftComplexT* ehat,
                             CufftComplexT* s11, CufftComplexT* s22,
                             CufftComplexT* s12,
                             int nxh, int ny,
                             double kx0, double ky0,
                             double C11, double C12, double C44,
                             double meanE11, double meanE22)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nxh * ny) return;
    const int ki = idx % nxh;
    int kj = idx / nxh;
    if (kj > ny / 2) kj -= ny;                 // negative frequencies

    if (ki == 0 && kj == 0) {                  // homogeneous (mean) strain
        const double n = 1.0;                  // ehat[0] = Σe* (unnormalised)
        s11[idx].x = meanE11 * ehat[idx].x * n;
        s11[idx].y = 0.0;
        s22[idx].x = meanE22 * ehat[idx].x * n;
        s22[idx].y = 0.0;
        s12[idx].x = 0.0;
        s12[idx].y = 0.0;
        return;
    }

    const double x1 = kx0 * ki;
    const double x2 = ky0 * kj;

    // acoustic tensor and its inverse
    const double K11 = C11 * x1 * x1 + C44 * x2 * x2;
    const double K22 = C11 * x2 * x2 + C44 * x1 * x1;
    const double K12 = (C12 + C44) * x1 * x2;
    const double det = K11 * K22 - K12 * K12;
    const double w1  = ( K22 * x1 - K12 * x2) / det;   // (K⁻¹ ξ)_1
    const double w2  = (-K12 * x1 + K11 * x2) / det;   // (K⁻¹ ξ)_2

    const double sRe = (C11 + C12) * ehat[idx].x;
    const double sIm = (C11 + C12) * ehat[idx].y;

    const double a11 = w1 * x1;
    const double a22 = w2 * x2;
    const double a12 = 0.5 * (w1 * x2 + w2 * x1);

    s11[idx].x = sRe * a11;  s11[idx].y = sIm * a11;
    s22[idx].x = sRe * a22;  s22[idx].y = sIm * a22;
    s12[idx].x = sRe * a12;  s12[idx].y = sIm * a12;
}

__global__ void kernel_energy(Real* out, const Real* e11, const Real* e22,
                              const Real* e12, const Real* estar,
                              Real C11, Real C12, Real C44,
                              int nx, int ny, int sx, int sy, int g)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx;
    const int j = idx / nx;
    const int c = (i + g) + sx * ((j + g) + sy * g);
    const Real d11 = e11[c] - estar[c];
    const Real d22 = e22[c] - estar[c];
    const Real d12 = e12[c];
    out[c] = Real(0.5) * (C11 * (d11 * d11 + d22 * d22)
                          + Real(2) * C12 * d11 * d22
                          + Real(4) * C44 * d12 * d12);
}

} // namespace

ElasticityFFT2D::ElasticityFFT2D(const Mesh& mesh, const ElasticParams2D& C)
    : mesh_(mesh), C_(C), nx_(mesh.n[0]), ny_(mesh.n[1])
{
    C_.validate();
    if (mesh.dim != 2)
        throw std::invalid_argument("ElasticityFFT2D: 2D meshes only");
    nc_ = (nx_ / 2 + 1) * ny_;

    CUFFT_CHECK(cufftPlan2d(&planF_, ny_, nx_, FFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&planB_, ny_, nx_, FFT_C2R));

    CUDA_CHECK(cudaMalloc(&d_real_, sizeof(Real) * nx_ * ny_));
    CUDA_CHECK(cudaMalloc(&d_hat_, sizeof(CufftComplexT) * nc_));
    CUDA_CHECK(cudaMalloc(&d_s11_, sizeof(CufftComplexT) * nc_));
    CUDA_CHECK(cudaMalloc(&d_s22_, sizeof(CufftComplexT) * nc_));
    CUDA_CHECK(cudaMalloc(&d_s12_, sizeof(CufftComplexT) * nc_));
    for (auto& p : d_out_)
        CUDA_CHECK(cudaMalloc(&p, sizeof(Real) * nx_ * ny_));
}

ElasticityFFT2D::~ElasticityFFT2D() {
    if (planF_) cufftDestroy(planF_);    // best-effort; no throw in dtor
    if (planB_) cufftDestroy(planB_);
    for (void* p : {static_cast<void*>(d_real_), d_hat_, d_s11_, d_s22_,
                    d_s12_, static_cast<void*>(d_out_[0]),
                    static_cast<void*>(d_out_[1]),
                    static_cast<void*>(d_out_[2])})
        if (p) cudaFree(p);
}

void ElasticityFFT2D::solve(const ScalarField& eStar,
                            ScalarField* e11, ScalarField* e22,
                            ScalarField* e12, ScalarField* elasticEnergy)
{
    if (!eStar.d_curr)
        throw std::runtime_error("ElasticityFFT2D::solve: eStar not on device");
    if (eStar.mesh.n[0] != nx_ || eStar.mesh.n[1] != ny_)
        throw std::invalid_argument("ElasticityFFT2D::solve: mesh mismatch");

    const int n = nx_ * ny_;
    const int blocks = (n + 255) / 256;
    const int cblocks = (nc_ + 255) / 256;

    kernel_pack<<<blocks, 256>>>(d_real_, eStar.d_curr, nx_, ny_,
                                 eStar.storedDims[0], eStar.storedDims[1],
                                 eStar.ghost);
    CUDA_CHECK(cudaGetLastError());

#ifdef PHIX_REAL_FLOAT
    CUFFT_CHECK(cufftExecR2C(planF_, d_real_,
                             static_cast<CufftComplexT*>(d_hat_)));
#else
    CUFFT_CHECK(cufftExecD2Z(planF_, d_real_,
                             static_cast<CufftComplexT*>(d_hat_)));
#endif

    // Mean-mode convention: ⟨σ⟩ = 0 → ⟨ε⟩ = ⟨ε*⟩ (dilatational e* only
    // couples to ε11 = ε22 = ⟨e*⟩); ⟨ε⟩ = 0 otherwise.
    const double meanFac = zeroMeanStress ? 1.0 : 0.0;

    kernel_modes<<<cblocks, 256>>>(
        static_cast<CufftComplexT*>(d_hat_),
        static_cast<CufftComplexT*>(d_s11_),
        static_cast<CufftComplexT*>(d_s22_),
        static_cast<CufftComplexT*>(d_s12_),
        nx_ / 2 + 1, ny_,
        2.0 * M_PI / (nx_ * mesh_.d[0]),
        2.0 * M_PI / (ny_ * mesh_.d[1]),
        C_.C11, C_.C12, C_.C44, meanFac, meanFac);
    CUDA_CHECK(cudaGetLastError());

    const Real scale = static_cast<Real>(1.0 / n);
    void* spectra[3] = {d_s11_, d_s22_, d_s12_};
    ScalarField* outs[3] = {e11, e22, e12};
    for (int q = 0; q < 3; ++q) {
#ifdef PHIX_REAL_FLOAT
        CUFFT_CHECK(cufftExecC2R(planB_,
                                 static_cast<CufftComplexT*>(spectra[q]),
                                 d_out_[q]));
#else
        CUFFT_CHECK(cufftExecZ2D(planB_,
                                 static_cast<CufftComplexT*>(spectra[q]),
                                 d_out_[q]));
#endif
        if (outs[q]) {
            if (!outs[q]->d_curr)
                throw std::runtime_error(
                    "ElasticityFFT2D::solve: output not on device");
            kernel_unpack<<<blocks, 256>>>(
                outs[q]->d_curr, d_out_[q], scale, nx_, ny_,
                outs[q]->storedDims[0], outs[q]->storedDims[1],
                outs[q]->ghost);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (elasticEnergy) {
        if (!e11 || !e22 || !e12)
            throw std::invalid_argument(
                "ElasticityFFT2D::solve: energy output needs all three "
                "strain outputs");
        kernel_energy<<<blocks, 256>>>(
            elasticEnergy->d_curr, e11->d_curr, e22->d_curr, e12->d_curr,
            eStar.d_curr,
            static_cast<Real>(C_.C11), static_cast<Real>(C_.C12),
            static_cast<Real>(C_.C44),
            nx_, ny_, elasticEnergy->storedDims[0],
            elasticEnergy->storedDims[1], elasticEnergy->ghost);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace PhiX
