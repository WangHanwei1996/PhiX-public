add_executable(module_field test_field.cpp)
target_link_libraries(module_field PRIVATE phix)
add_test(NAME module_field COMMAND module_field)

add_executable(module_reduce test_reduce.cu)
target_link_libraries(module_reduce PRIVATE phix)
set_target_properties(module_reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_reduce COMMAND module_reduce)

add_executable(module_timelevel test_timelevel.cu)
target_link_libraries(module_timelevel PRIVATE phix)
set_target_properties(module_timelevel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_timelevel COMMAND module_timelevel)

add_executable(module_energy test_energy.cu)
target_link_libraries(module_energy PRIVATE phix)
set_target_properties(module_energy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_energy COMMAND module_energy)

add_executable(module_storage test_storage.cpp)
target_link_libraries(module_storage PRIVATE phix)
add_test(NAME module_storage COMMAND module_storage)

add_executable(module_indexfield test_indexfield.cu)
target_link_libraries(module_indexfield PRIVATE phix)
set_target_properties(module_indexfield PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_indexfield COMMAND module_indexfield)

add_executable(module_mask test_mask.cu)
target_link_libraries(module_mask PRIVATE phix)
set_target_properties(module_mask PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_mask COMMAND module_mask)
