add_executable(bench_stencil bench_stencil.cu)
target_link_libraries(bench_stencil PRIVATE phix)
set_target_properties(bench_stencil PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME bench_stencil COMMAND bench_stencil)

add_executable(bench_semiimplicit bench_semiimplicit.cu)
target_link_libraries(bench_semiimplicit PRIVATE phix)
set_target_properties(bench_semiimplicit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME bench_semiimplicit COMMAND bench_semiimplicit)

# Optional CUPTI counts actual runtime kernel launches; timing runs without callbacks.
add_executable(bench_numerics bench_numerics.cu)
target_link_libraries(bench_numerics PRIVATE phix)
set_target_properties(bench_numerics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
find_path(PHIX_CUPTI_INCLUDE cupti.h HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/include" "/usr/local/cuda/extras/CUPTI/include")
find_library(PHIX_CUPTI_LIBRARY cupti HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/lib64" "/usr/local/cuda/extras/CUPTI/lib64")
if(PHIX_CUPTI_INCLUDE AND PHIX_CUPTI_LIBRARY)
    target_include_directories(bench_numerics PRIVATE "${PHIX_CUPTI_INCLUDE}")
    target_link_libraries(bench_numerics PRIVATE "${PHIX_CUPTI_LIBRARY}")
    target_compile_definitions(bench_numerics PRIVATE PHIX_HAVE_CUPTI)
endif()

# Setup/JIT is excluded from this explicit performance comparison.
add_executable(bench_symbolic bench_symbolic.cu)
target_link_libraries(bench_symbolic PRIVATE phix)
set_target_properties(bench_symbolic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
