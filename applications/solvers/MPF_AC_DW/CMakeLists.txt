# [makePhi generated]

add_executable(MPF_AC_DW MPF_AC_DW.cu)

target_link_libraries(MPF_AC_DW PRIVATE phix)

set_target_properties(MPF_AC_DW PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
