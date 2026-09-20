// ---------------------------------------------------------------------------
// module_perf — perf/Perf.h instrumentation primitives
// ---------------------------------------------------------------------------

#include "perf/Perf.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thread>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // WallTimer measures a 50 ms sleep (generous bounds — CI jitter)
    {
        perf::WallTimer t;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const double s = t.seconds();
        require(s >= 0.04 && s < 1.0,
                "WallTimer out of range: " + std::to_string(s));
        t.reset();
        require(t.seconds() < 0.04, "WallTimer reset did not restart");
    }

    // CudaEventTimer times real device work (10 × 64 MB memset)
    {
        const std::size_t bytes = 64u << 20;
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");

        perf::CudaEventTimer t;
        t.start();
        for (int i = 0; i < 10; ++i) cudaMemset(d, 0, bytes);
        const double ms = t.stopMs();
        cudaFree(d);

        require(ms > 0.0 && ms < 5000.0,
                "CudaEventTimer out of range: " + std::to_string(ms));
    }

    // NVTX range must compile and be scope-safe in both build modes
    {
        PHIX_NVTX_RANGE("module_perf/outer");
        {
            PHIX_NVTX_RANGE("module_perf/inner");
        }
    }

    return 0;
}
