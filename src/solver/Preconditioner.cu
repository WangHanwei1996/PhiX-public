#include "solver/Preconditioner.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

inline int blocks(std::size_t n) { return static_cast<int>((n + 255) / 256); }

// ---------------------------------------------------------------------------
// Diagonal: z = r / (alpha − sigma·diagL)
// ---------------------------------------------------------------------------
__global__ void k_diag_const(Real* z, const Real* r, const double* alpha,
                             const double* sigma, double diagL, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = r[i] / static_cast<Real>(*alpha - *sigma * diagL);
}

__global__ void k_diag_field(Real* z, const Real* r, const double* alpha,
                             const double* sigma, const Real* diagL, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = r[i] / (static_cast<Real>(*alpha)
                       - static_cast<Real>(*sigma) * diagL[i]);
}

// ---------------------------------------------------------------------------
// Chebyshev recurrence slots:
//   c[0] = sigma1 = theta/delta      c[1] = rho
//   c[2] = coefA  = rhoNew·rhoOld    c[3] = coefB = 2·rhoNew/delta
//   c[4] = 1/theta                   c[5] = delta
// Interval [a, b] = [alpha, alpha + sigma·lamMaxL], theta = (b+a)/2,
// delta = (b−a)/2 — computed on device from the CG slots each apply.
// ---------------------------------------------------------------------------
__global__ void k_cheb_init(double* c, const double* alpha,
                            const double* sigma, double lamMaxL)
{
    const double a     = *alpha;
    const double b     = *alpha + *sigma * lamMaxL;
    const double theta = 0.5 * (b + a);
    const double delta = 0.5 * (b - a);
    c[0] = theta / delta;
    c[1] = delta / theta;      // rho_1 = 1/sigma1
    c[4] = 1.0 / theta;
    c[5] = delta;
}

__global__ void k_cheb_rho(double* c)
{
    const double rhoNew = 1.0 / (2.0 * c[0] - c[1]);
    c[2] = rhoNew * c[1];
    c[3] = 2.0 * rhoNew / c[5];
    c[1] = rhoNew;
}

// z_1 = (1/theta)·r ;  d_1 = z_1
__global__ void k_cheb_start(Real* z, Real* d, const Real* r,
                             const double* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real v = static_cast<Real>(c[4]) * r[i];
        z[i] = v;
        d[i] = v;
    }
}

// s = r − A·z = r − alpha·z + sigma·Lz ;  d = coefA·d + coefB·s ;  z += d
__global__ void k_cheb_update(Real* z, Real* d, const Real* r, const Real* Lz,
                              const double* c, const double* alpha,
                              const double* sigma, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real s = r[i] - static_cast<Real>(*alpha) * z[i]
                     + static_cast<Real>(*sigma) * Lz[i];
        const Real dn = static_cast<Real>(c[2]) * d[i]
                      + static_cast<Real>(c[3]) * s;
        d[i] = dn;
        z[i] += dn;
    }
}

ScalarField makeScratch(const Mesh& mesh, int ghost, const char* name) {
    ScalarField f(mesh, name, ghost);
    f.allocDevice();
    CUDA_CHECK(cudaMemset(f.d_curr, 0, f.storedSize * sizeof(Real)));
    return f;
}

} // namespace

// ===========================================================================
// DiagonalPreconditioner
// ===========================================================================

DiagonalPreconditioner::DiagonalPreconditioner(double diagL)
    : diagConst_(diagL)
{
    if (diagL > 0.0)
        throw std::invalid_argument(
            "DiagonalPreconditioner: diag(L) must be <= 0 (SPD A)");
}

DiagonalPreconditioner::DiagonalPreconditioner(const ScalarField* diagL)
    : diagField_(diagL)
{
    if (!diagL || !diagL->d_curr)
        throw std::invalid_argument(
            "DiagonalPreconditioner: diag field must be device-resident");
}

void DiagonalPreconditioner::apply(ScalarField& r, ScalarField& z,
                                   const double* d_alpha,
                                   const double* d_sigma,
                                   cudaStream_t stream)
{
    const int n = static_cast<int>(r.storedSize);
    if (diagField_) {
        k_diag_field<<<blocks(n), 256, 0, stream>>>(
            z.d_curr, r.d_curr, d_alpha, d_sigma, diagField_->d_curr, n);
    } else {
        k_diag_const<<<blocks(n), 256, 0, stream>>>(
            z.d_curr, r.d_curr, d_alpha, d_sigma, diagConst_, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// ChebyshevPreconditioner
// ===========================================================================

ChebyshevPreconditioner::ChebyshevPreconditioner(LinearOperator& L,
                                                 double lamMaxL, int degree,
                                                 const Mesh& mesh, int ghost)
    : L_(L)
    , lamMaxL_(lamMaxL)
    , degree_(degree)
    , t_(makeScratch(mesh, ghost, "_cheb_t"))
    , d_(makeScratch(mesh, ghost, "_cheb_d"))
{
    if (lamMaxL <= 0.0)
        throw std::invalid_argument(
            "ChebyshevPreconditioner: lamMaxL must be > 0");
    if (degree < 1)
        throw std::invalid_argument(
            "ChebyshevPreconditioner: degree must be >= 1");
    if (ghost < L.ghostRequired())
        throw std::invalid_argument(
            "ChebyshevPreconditioner: ghost < operator requirement");
    CUDA_CHECK(cudaMalloc(&d_c_, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_c_, 0, 6 * sizeof(double)));
}

ChebyshevPreconditioner::~ChebyshevPreconditioner() {
    if (d_c_) cudaFree(d_c_);   // best-effort; no throw in dtor
}

double ChebyshevPreconditioner::lambdaMaxCD2(const Mesh& mesh, double D,
                                             double shift)
{
    double s = 0.0;
    for (int ax = 0; ax < mesh.dim; ++ax)
        s += 1.0 / (mesh.d[ax] * mesh.d[ax]);
    return 4.0 * D * s + shift;
}

void ChebyshevPreconditioner::apply(ScalarField& r, ScalarField& z,
                                    const double* d_alpha,
                                    const double* d_sigma,
                                    cudaStream_t stream)
{
    const int n = static_cast<int>(r.storedSize);
    k_cheb_init<<<1, 1, 0, stream>>>(d_c_, d_alpha, d_sigma, lamMaxL_);
    CUDA_CHECK(cudaGetLastError());
    k_cheb_start<<<blocks(n), 256, 0, stream>>>(z.d_curr, d_.d_curr,
                                                r.d_curr, d_c_, n);
    CUDA_CHECK(cudaGetLastError());
    for (int k = 2; k <= degree_; ++k) {
        L_.apply(z, t_, stream);                     // Lz (refreshes z ghosts)
        k_cheb_rho<<<1, 1, 0, stream>>>(d_c_);
        k_cheb_update<<<blocks(n), 256, 0, stream>>>(
            z.d_curr, d_.d_curr, r.d_curr, t_.d_curr,
            d_c_, d_alpha, d_sigma, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace PhiX
