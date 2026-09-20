add_executable(quickstart quickstart.cu)

target_link_libraries(quickstart PRIVATE phix)

set_target_properties(quickstart PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # The binary lands next to the source so that ./quickstart finds
    # settings/ and writes output/ here, not inside build/.
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
