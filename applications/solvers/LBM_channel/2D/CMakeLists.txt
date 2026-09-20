# [makePhi generated]

add_executable(LBM_channel LBM_channel.cu)

target_link_libraries(LBM_channel PRIVATE phix)

set_target_properties(LBM_channel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
