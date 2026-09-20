# [makePhi generated]

add_executable(CH_elastic_precipitate CH_elastic_precipitate.cu)

target_link_libraries(CH_elastic_precipitate PRIVATE phix)

set_target_properties(CH_elastic_precipitate PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
