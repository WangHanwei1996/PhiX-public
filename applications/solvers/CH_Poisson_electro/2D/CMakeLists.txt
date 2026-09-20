# [makePhi generated]

add_executable(CH_Poisson_electro CH_Poisson_electro.cu)

target_link_libraries(CH_Poisson_electro PRIVATE phix)

set_target_properties(CH_Poisson_electro PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
