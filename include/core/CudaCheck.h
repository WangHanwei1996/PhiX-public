#pragma once

// ---------------------------------------------------------------------------
// CudaCheck.h — central CUDA error checking.  Include from nvcc-compiled
// TUs (.cu) and CUDA-touching headers/.inl only — it pulls in
// <cuda_runtime.h>.
//
//   PHIX_CUDA_CHECK(cudaMalloc(&p, bytes));   // any cudaError_t-returning call
//   kernel<<<grd, blk>>>(...);
//   PHIX_KERNEL_CHECK("Module::kernelName");  // cudaGetLastError after launch
//
// Both throw PhiX::DeviceError (core/Error.h) with the failing expression,
// file:line, and the CUDA error name + description.
//
// CUDA_CHECK is kept as an alias of PHIX_CUDA_CHECK: the historical
// per-file `#define CUDA_CHECK` copies used that name, so consolidating a
// file is just "delete the local #define, include this header" with zero
// call-site churn.
//
// Destructor cleanup calls (cudaFree in dtors) stay deliberately
// unchecked — never throw from a destructor.
// ---------------------------------------------------------------------------

#include "core/Error.h"

#include <cuda_runtime.h>

#include <string>

namespace PhiX {
namespace detail {

[[noreturn]] inline void throwCuda(cudaError_t e, const char* expr,
                                   const char* file, int line) {
    throw DeviceError(
        std::string(file) + ":" + std::to_string(line),
        std::string(cudaGetErrorName(e)) + " (" + cudaGetErrorString(e)
            + ") in " + expr,
        "check device memory, launch configuration, and pointer lifetimes; "
        "run under compute-sanitizer for the exact fault site");
}

// Check the sticky CUDA error state after a kernel launch.
// `who` names the launching function for the error message.
inline void checkKernelLaunch(const char* who,
                              const char* file = nullptr, int line = 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::string where = who ? who : "kernel launch";
        if (file)
            where += std::string(" (") + file + ":" + std::to_string(line) + ")";
        throw DeviceError(
            where,
            std::string(cudaGetErrorName(e)) + " (" + cudaGetErrorString(e)
                + ") after kernel launch",
            "grid/block dimensions, device pointers, and the GPU arch the "
            "binary was built for (PHIX_CUDA_ARCH) are the usual suspects");
    }
}

} // namespace detail
} // namespace PhiX

#define PHIX_CUDA_CHECK(call)                                                  \
    do {                                                                       \
        cudaError_t _phix_e = (call);                                          \
        if (_phix_e != cudaSuccess)                                            \
            ::PhiX::detail::throwCuda(_phix_e, #call, __FILE__, __LINE__);     \
    } while (0)

#define PHIX_KERNEL_CHECK(who)                                                 \
    ::PhiX::detail::checkKernelLaunch((who), __FILE__, __LINE__)

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) PHIX_CUDA_CHECK(call)
#endif
