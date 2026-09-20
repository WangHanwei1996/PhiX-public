#include "field/NoiseGenerator.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

// ===========================================================================
// CUDA kernels
//
// Thread layout: 1D.
//   tid = blockIdx.x * blockDim.x + threadIdx.x
//       = iz * ny * nx + iy * nx + ix   (physical-cell linear index)
//
// Stored index for physical cell (ix, iy, iz):
//   (ix + ghost) + sx * ((iy + ghost) + sy * (iz + ghost))
// ===========================================================================

// ---------------------------------------------------------------------------
// State initialisation
// ---------------------------------------------------------------------------
__global__ void k_noise_init_states(curandState* states,
                                     unsigned long long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// ---------------------------------------------------------------------------
// GAUSSIAN  d_curr[idx] += N(mean, std_dev^2)
// ---------------------------------------------------------------------------
__global__ void k_noise_gaussian(Real* d_curr, curandState* states,
                                  double mean, double std_dev,
                                  int n, int nx, int ny,
                                  int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    d_curr[idx] += mean + std_dev * curand_normal_double(&states[tid]);
}

// ---------------------------------------------------------------------------
// UNIFORM  d_curr[idx] += U[lo, hi]
// curand_uniform_double returns (0, 1].
// ---------------------------------------------------------------------------
__global__ void k_noise_uniform(Real* d_curr, curandState* states,
                                 double lo, double hi,
                                 int n, int nx, int ny,
                                 int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    d_curr[idx] += lo + (hi - lo) * curand_uniform_double(&states[tid]);
}

// ---------------------------------------------------------------------------
// LOG_NORMAL  d_curr[idx] += LogNormal(mu, sigma)
// curand_log_normal_double(state, mu, sigma) generates X = exp(mu + sigma*Z)
// where Z ~ N(0,1), so X ~ LogNormal with underlying N(mu, sigma^2).
// ---------------------------------------------------------------------------
__global__ void k_noise_log_normal(Real* d_curr, curandState* states,
                                    double mu, double sigma,
                                    int n, int nx, int ny,
                                    int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    d_curr[idx] += curand_log_normal_double(&states[tid], mu, sigma);
}

// ---------------------------------------------------------------------------
// CAUCHY  d_curr[idx] += Cauchy(location, scale)
// Inverse CDF: X = location + scale * tan(pi * (U - 0.5))
// where U ~ Uniform(0, 1].
// ---------------------------------------------------------------------------
__global__ void k_noise_cauchy(Real* d_curr, curandState* states,
                                double location, double scale,
                                int n, int nx, int ny,
                                int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    double u = curand_uniform_double(&states[tid]);
    d_curr[idx] += location + scale * tan(3.14159265358979323846 * (u - 0.5));
}

// ---------------------------------------------------------------------------
// EXPONENTIAL  d_curr[idx] += shift + Exponential(scale)
// Inverse CDF: X = -scale * log(U), where U ~ Uniform(0, 1].
// Final sample = shift - scale * log(U)  (always >= shift).
// ---------------------------------------------------------------------------
__global__ void k_noise_exponential(Real* d_curr, curandState* states,
                                     double shift, double scale,
                                     int n, int nx, int ny,
                                     int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    double u = curand_uniform_double(&states[tid]);
    d_curr[idx] += shift - scale * log(u);
}

// ---------------------------------------------------------------------------
// BERNOULLI  d_curr[idx] += (U <= p) ? +amplitude : -amplitude
// ---------------------------------------------------------------------------
__global__ void k_noise_bernoulli(Real* d_curr, curandState* states,
                                   double amplitude, double p,
                                   int n, int nx, int ny,
                                   int sx, int sy, int g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ix = tid % nx;
    int iy = (tid / nx) % ny;
    int iz = tid / (nx * ny);
    int idx = (ix + g) + sx * ((iy + g) + sy * (iz + g));
    double u = curand_uniform_double(&states[tid]);
    d_curr[idx] += (u <= p) ? amplitude : -amplitude;
}

// ===========================================================================
// NoiseGenerator implementation
// ===========================================================================

NoiseGenerator::NoiseGenerator(const ScalarField& ref,
                                unsigned long long seed)
    : nPhys_(ref.mesh.n[0] * ref.mesh.n[1] * ref.mesh.n[2])
    , nx_(ref.mesh.n[0]), ny_(ref.mesh.n[1]), nz_(ref.mesh.n[2])
    , sx_(ref.storedDims[0]), sy_(ref.storedDims[1])
    , ghost_(ref.ghost)
{
    CUDA_CHECK(cudaMalloc(&d_states_,
                          static_cast<std::size_t>(nPhys_) * sizeof(curandState)));
    reseed(seed);
}

NoiseGenerator::~NoiseGenerator() {
    if (d_states_) {
        cudaFree(d_states_);
        d_states_ = nullptr;
    }
}

NoiseGenerator::NoiseGenerator(NoiseGenerator&& other) noexcept
    : d_states_(other.d_states_)
    , nPhys_(other.nPhys_)
    , nx_(other.nx_), ny_(other.ny_), nz_(other.nz_)
    , sx_(other.sx_), sy_(other.sy_)
    , ghost_(other.ghost_)
{
    other.d_states_ = nullptr;
}

NoiseGenerator& NoiseGenerator::operator=(NoiseGenerator&& other) noexcept {
    if (this == &other) return *this;
    if (d_states_) cudaFree(d_states_);
    d_states_ = other.d_states_;  other.d_states_ = nullptr;
    nPhys_  = other.nPhys_;
    nx_     = other.nx_;   ny_    = other.ny_;   nz_    = other.nz_;
    sx_     = other.sx_;   sy_    = other.sy_;
    ghost_  = other.ghost_;
    return *this;
}

void NoiseGenerator::reseed(unsigned long long seed) {
    const int threads = 256;
    const int blocks  = (nPhys_ + threads - 1) / threads;
    k_noise_init_states<<<blocks, threads>>>(d_states_, seed, nPhys_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void NoiseGenerator::apply(ScalarField& f, NoiseType type,
                            double param1, double param2)
{
    if (!f.d_curr)
        throw std::runtime_error(
            "NoiseGenerator::apply: field not on device — "
            "call allocDevice() and uploadToDevice() first");

    const int threads = 256;
    const int blocks  = (nPhys_ + threads - 1) / threads;

    switch (type) {
    case NoiseType::GAUSSIAN:
        k_noise_gaussian<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    case NoiseType::UNIFORM:
        k_noise_uniform<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    case NoiseType::LOG_NORMAL:
        k_noise_log_normal<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    case NoiseType::CAUCHY:
        k_noise_cauchy<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    case NoiseType::EXPONENTIAL:
        k_noise_exponential<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    case NoiseType::BERNOULLI:
        k_noise_bernoulli<<<blocks, threads>>>(
            f.d_curr, d_states_,
            param1, param2,
            nPhys_, nx_, ny_, sx_, sy_, ghost_);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
