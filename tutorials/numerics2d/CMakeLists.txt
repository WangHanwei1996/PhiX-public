add_executable(numerics2d numerics2d.cu)
target_link_libraries(numerics2d PRIVATE phix)
set_target_properties(numerics2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/run")
add_test(NAME sample_numerics2d COMMAND numerics2d "${CMAKE_CURRENT_SOURCE_DIR}/settings/settings.jsonc")
set_tests_properties(sample_numerics2d PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/run")
