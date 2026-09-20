#pragma once

// ---------------------------------------------------------------------------
// Perf.h — lightweight performance instrumentation.
//
//   WallTimer       host wall-clock stopwatch (steady_clock)
//   CudaEventTimer  device-side interval timing via cudaEvent (per stream)
//   PHIX_NVTX_RANGE("label")  scoped NVTX range for Nsight Systems timelines
//
// NVTX ranges compile to no-ops unless the build sets -DPHIX_ENABLE_NVTX
// (CMake option PHIX_ENABLE_NVTX=ON).  Combine with the PHIX_LINEINFO=ON
// option (-lineinfo) for source-correlated kernel profiling.
//
// This header needs the CUDA toolkit include path (cuda_runtime.h) — include
// it from .cu files, or add the toolkit include dir for plain .cpp.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

#ifdef PHIX_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace PhiX {
namespace perf {

// ---------------------------------------------------------------------------
// WallTimer — host stopwatch.  Running after construction; restart with
// reset(); seconds() reads without stopping.
// ---------------------------------------------------------------------------
class WallTimer {
public:
    WallTimer() : t0_(Clock::now()) {}

    void reset() { t0_ = Clock::now(); }

    double seconds() const {
        return std::chrono::duration<double>(Clock::now() - t0_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_;
};

// ---------------------------------------------------------------------------
// CudaEventTimer — measures GPU time between start() and stop() on a stream.
// stop() synchronises on the stop event, so the returned time is final.
//
//   CudaEventTimer t;
//   t.start();          // optionally t.start(stream)
//   ... kernel launches ...
//   double ms = t.stopMs();
// ---------------------------------------------------------------------------
class CudaEventTimer {
public:
    CudaEventTimer() {
        check_(cudaEventCreate(&begin_), "cudaEventCreate");
        check_(cudaEventCreate(&end_),   "cudaEventCreate");
    }
    ~CudaEventTimer() {
        cudaEventDestroy(begin_);   // best-effort; no throw in dtor
        cudaEventDestroy(end_);
    }
    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr) {
        check_(cudaEventRecord(begin_, stream), "cudaEventRecord(begin)");
    }

    // Records the stop event, waits for it, returns elapsed milliseconds.
    double stopMs(cudaStream_t stream = nullptr) {
        check_(cudaEventRecord(end_, stream), "cudaEventRecord(end)");
        check_(cudaEventSynchronize(end_),    "cudaEventSynchronize");
        float ms = 0.0f;
        check_(cudaEventElapsedTime(&ms, begin_, end_), "cudaEventElapsedTime");
        return static_cast<double>(ms);
    }

private:
    static void check_(cudaError_t e, const char* what) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CudaEventTimer: ") + what
                                     + ": " + cudaGetErrorString(e));
    }
    cudaEvent_t begin_{}, end_{};
};

// ---------------------------------------------------------------------------
// NvtxRange — RAII NVTX range (no-op unless PHIX_ENABLE_NVTX)
// ---------------------------------------------------------------------------
class NvtxRange {
public:
#ifdef PHIX_ENABLE_NVTX
    explicit NvtxRange(const char* label) { nvtxRangePushA(label); }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char*) {}
#endif
    NvtxRange(const NvtxRange&)            = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace perf
} // namespace PhiX

// Scoped NVTX range with a unique local name
#define PHIX_NVTX_CONCAT_(a, b) a##b
#define PHIX_NVTX_CONCAT(a, b) PHIX_NVTX_CONCAT_(a, b)
#define PHIX_NVTX_RANGE(label) \
    ::PhiX::perf::NvtxRange PHIX_NVTX_CONCAT(phix_nvtx_range_, __LINE__)(label)
