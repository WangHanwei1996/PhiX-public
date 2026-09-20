# [makePhi generated]

add_executable(dendrite_growth dendrite_growth.cu)

target_link_libraries(dendrite_growth PRIVATE phix)

set_target_properties(dendrite_growth PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
