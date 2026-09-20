# [makePhi generated]

add_executable(Cahn-Hillard_double-well Cahn-Hillard_double-well.cu)

target_link_libraries(Cahn-Hillard_double-well PRIVATE phix)

set_target_properties(Cahn-Hillard_double-well PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
