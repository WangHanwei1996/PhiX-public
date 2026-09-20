add_executable(module_error test_error.cpp)
target_link_libraries(module_error PRIVATE phix)
add_test(NAME module_error COMMAND module_error)

add_executable(module_check test_check.cu)
target_link_libraries(module_check PRIVATE phix)
set_target_properties(module_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_check COMMAND module_check)

add_executable(module_misuse test_misuse.cu)
target_link_libraries(module_misuse PRIVATE phix)
set_target_properties(module_misuse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_misuse COMMAND module_misuse)

add_executable(module_io_misuse test_io_misuse.cu)
target_link_libraries(module_io_misuse PRIVATE phix)
set_target_properties(module_io_misuse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_io_misuse COMMAND module_io_misuse)

add_executable(module_health test_health.cu)
target_link_libraries(module_health PRIVATE phix)
set_target_properties(module_health PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_health COMMAND module_health)
