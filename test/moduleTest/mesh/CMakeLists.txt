add_executable(module_mesh test_mesh.cpp)
target_link_libraries(module_mesh PRIVATE phix)
add_test(NAME module_mesh COMMAND module_mesh)
