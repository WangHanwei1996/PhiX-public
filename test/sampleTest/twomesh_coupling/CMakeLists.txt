add_executable(twomesh_coupling twomesh_coupling.cu)

target_link_libraries(twomesh_coupling PRIVATE phix)

set_target_properties(twomesh_coupling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME sample_twomesh_coupling COMMAND twomesh_coupling)
