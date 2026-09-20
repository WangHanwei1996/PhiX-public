add_executable(float_smoke test_float_smoke.cu)
target_link_libraries(float_smoke PRIVATE phix)
set_target_properties(float_smoke PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_smoke COMMAND float_smoke)

add_executable(float_scheme_catalog ../moduleTest/scheme/test_catalog.cu)
target_link_libraries(float_scheme_catalog PRIVATE phix)
set_target_properties(float_scheme_catalog PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_scheme_catalog COMMAND float_scheme_catalog)

add_executable(float_numerics ../moduleTest/numerics/test_numerics.cu)
target_link_libraries(float_numerics PRIVATE phix)
set_target_properties(float_numerics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_numerics COMMAND float_numerics)

add_executable(float_numerics_extensions ../moduleTest/numerics/test_extensions.cu)
target_link_libraries(float_numerics_extensions PRIVATE phix)
set_target_properties(float_numerics_extensions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_numerics_extensions COMMAND float_numerics_extensions)

add_executable(float_symbolic ../moduleTest/numerics/test_symbolic.cu)
target_link_libraries(float_symbolic PRIVATE phix)
set_target_properties(float_symbolic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_symbolic COMMAND float_symbolic)

add_executable(float_symbolic_system ../moduleTest/numerics/test_symbolic_system.cu)
target_link_libraries(float_symbolic_system PRIVATE phix)
set_target_properties(float_symbolic_system PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_symbolic_system COMMAND float_symbolic_system)

add_executable(float_composite ../moduleTest/numerics/test_composite.cu)
target_link_libraries(float_composite PRIVATE phix)
set_target_properties(float_composite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_composite COMMAND float_composite)

add_executable(float_normal_vector ../moduleTest/numerics/test_normal_vector.cu)
target_link_libraries(float_normal_vector PRIVATE phix)
set_target_properties(float_normal_vector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME float_normal_vector COMMAND float_normal_vector)
