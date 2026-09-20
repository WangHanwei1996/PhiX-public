# [makePhi generated]

add_executable(diffusion_1D diffusion_1D.cu)

target_link_libraries(diffusion_1D PRIVATE phix)

set_target_properties(diffusion_1D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(diffusion_compare diffusion_compare.cu)

target_link_libraries(diffusion_compare PRIVATE phix)

set_target_properties(diffusion_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
