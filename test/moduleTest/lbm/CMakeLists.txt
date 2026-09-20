add_executable(module_lbm test_lbm.cu)
target_link_libraries(module_lbm PRIVATE phix)
set_target_properties(module_lbm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_lbm COMMAND module_lbm)
