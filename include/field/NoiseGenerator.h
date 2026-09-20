#pragma once

#include "field/ScalarField.h"

#include <curand_kernel.h>

namespace PhiX {

// ---------------------------------------------------------------------------
// NoiseType — probability distribution used for noise injection.
//
//   GAUSSIAN    : additive N(param1, param2^2)
//                 param1 = mean,          param2 = std_dev
//
//   UNIFORM     : additive U[param1, param2]
//                 param1 = lo,            param2 = hi
//
//   LOG_NORMAL  : additive LogNormal with underlying N(param1, param2^2)
//                 param1 = mu  (mean of the underlying normal distribution)
//                 param2 = sigma (std of the underlying normal distribution)
//                 Sample is always positive; useful for multiplicative noise.
//
//   CAUCHY      : additive Cauchy(param1, param2)  (heavy-tailed, Levy-stable)
//                 param1 = location,      param2 = scale
//                 Inverse CDF: location + scale * tan(pi * (U - 0.5))
//
//   EXPONENTIAL : additive shifted Exponential
//                 param1 = location shift, param2 = scale (= 1/rate)
//                 Sample = param1 - param2 * log(U)
//                 Always >= param1; useful for one-sided stochastic forcing.
//
//   BERNOULLI   : additive +/-amplitude with probability p / (1-p)
//                 param1 = amplitude,     param2 = p (probability of +amplitude)
//                 Useful for discrete salt-and-pepper perturbations.
// ---------------------------------------------------------------------------
enum class NoiseType {
    GAUSSIAN,
    UNIFORM,
    LOG_NORMAL,
    CAUCHY,
    EXPONENTIAL,
    BERNOULLI,
};

// ---------------------------------------------------------------------------
// NoiseGenerator
//
// Manages one curandStateXORWOW per physical cell and provides in-place GPU
// noise injection into ScalarField::d_curr.
//
// Noise is *additive*: d_curr[idx] += sample.
// No clamping is performed — apply a separate clamp step if needed.
//
// The generator is bound to the mesh layout (nx, ny, nz, storedDims, ghost)
// captured at construction from `ref`.  Any ScalarField sharing the same
// layout (same mesh and ghost width) can be passed to apply().
//
// Thread safety: not thread-safe; use one NoiseGenerator per stream/thread.
//
// Synchronisation:
//   apply()  — does NOT call cudaDeviceSynchronize.
//   reseed() — DOES call cudaDeviceSynchronize (state init must complete).
// ---------------------------------------------------------------------------
class NoiseGenerator {
public:
    // Construct from a reference field (captures mesh layout) and initial seed.
    explicit NoiseGenerator(const ScalarField& ref,
                            unsigned long long seed = 42ULL);

    ~NoiseGenerator();

    // Non-copyable (owns device memory)
    NoiseGenerator(const NoiseGenerator&)            = delete;
    NoiseGenerator& operator=(const NoiseGenerator&) = delete;

    // Movable
    NoiseGenerator(NoiseGenerator&&) noexcept;
    NoiseGenerator& operator=(NoiseGenerator&&) noexcept;

    // -----------------------------------------------------------------------
    // apply — add noise in-place to f.d_curr.
    //
    // f must have d_curr allocated and share the same mesh layout as `ref`.
    // Does NOT call cudaDeviceSynchronize.
    // -----------------------------------------------------------------------
    void apply(ScalarField& f, NoiseType type,
               double param1, double param2);

    // -----------------------------------------------------------------------
    // reseed — reinitialise curand states with a new seed.
    // Calls cudaDeviceSynchronize() before returning.
    // -----------------------------------------------------------------------
    void reseed(unsigned long long seed);

private:
    curandState* d_states_ = nullptr;
    int nPhys_;        // nx * ny * nz (number of physical cells)
    int nx_, ny_, nz_;
    int sx_, sy_;      // storedDims[0], storedDims[1]
    int ghost_;
};

} // namespace PhiX
