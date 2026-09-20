add_executable(module_material_perf test_free_energy_perf.cu)
target_link_libraries(module_material_perf PRIVATE phix)
set_target_properties(module_material_perf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_material_perf COMMAND module_material_perf)

add_executable(module_kks test_kks.cu)
target_link_libraries(module_kks PRIVATE phix)
set_target_properties(module_kks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_kks COMMAND module_kks)

add_executable(module_kks_at test_kks_antitrapping.cu)
target_link_libraries(module_kks_at PRIVATE phix)
set_target_properties(module_kks_at PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_kks_at COMMAND module_kks_at)

add_executable(module_interpolants test_interpolants.cu)
target_link_libraries(module_interpolants PRIVATE phix)
set_target_properties(module_interpolants PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_interpolants COMMAND module_interpolants)

add_executable(module_gfa_thermo test_gfa_thermo.cu)
target_link_libraries(module_gfa_thermo PRIVATE phix)
set_target_properties(module_gfa_thermo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_gfa_thermo
         COMMAND module_gfa_thermo ${CMAKE_SOURCE_DIR}/data/material_properties/Fe-B)

add_executable(module_parabolic_fit test_parabolic_fit.cu)
target_link_libraries(module_parabolic_fit PRIVATE phix)
set_target_properties(module_parabolic_fit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_parabolic_fit
         COMMAND module_parabolic_fit ${CMAKE_SOURCE_DIR}/data/material_properties/Fe-B)
