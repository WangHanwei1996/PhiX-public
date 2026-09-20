# [makePhi generated]

add_executable(CH_4AC_Ostwald CH_4AC_Ostwald.cu)

target_link_libraries(CH_4AC_Ostwald PRIVATE phix)

set_target_properties(CH_4AC_Ostwald PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
