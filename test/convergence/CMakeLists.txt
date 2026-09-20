add_executable(conv_spatial test_spatial_order.cu)
target_link_libraries(conv_spatial PRIVATE phix)
set_target_properties(conv_spatial PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME conv_spatial COMMAND conv_spatial)

add_executable(conv_temporal test_temporal_order.cu)
target_link_libraries(conv_temporal PRIVATE phix)
set_target_properties(conv_temporal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME conv_temporal COMMAND conv_temporal)

add_executable(conv_pde test_pde_convergence.cu)
target_link_libraries(conv_pde PRIVATE phix)
set_target_properties(conv_pde PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME conv_pde COMMAND conv_pde)

add_executable(conv_advection test_advection_order.cu)
target_link_libraries(conv_advection PRIVATE phix)
set_target_properties(conv_advection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME conv_advection COMMAND conv_advection)
