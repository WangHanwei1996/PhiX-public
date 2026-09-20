add_executable(equations2d equations2d.cu)
target_link_libraries(equations2d PRIVATE phix)
set_target_properties(equations2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/run")
add_test(NAME sample_equations2d COMMAND equations2d
    "${PROJECT_SOURCE_DIR}/etc/caseTemplates/equations2d/nonlinear_diffusion/settings/settings.jsonc")
set_tests_properties(sample_equations2d PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/run")
