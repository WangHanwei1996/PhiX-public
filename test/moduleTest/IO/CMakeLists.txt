add_executable(module_io_vtk test_vtk_legacy.cpp)
target_link_libraries(module_io_vtk PRIVATE phix)
add_test(NAME module_io_vtk COMMAND module_io_vtk)

add_executable(module_io_vti test_vti.cpp)
target_link_libraries(module_io_vti PRIVATE phix)
add_test(NAME module_io_vti COMMAND module_io_vti)

add_executable(module_io_diaglog test_diaglog.cpp)
target_link_libraries(module_io_diaglog PRIVATE phix)
add_test(NAME module_io_diaglog COMMAND module_io_diaglog)

add_executable(module_io_vts test_vts.cpp)
target_link_libraries(module_io_vts PRIVATE phix)
add_test(NAME module_io_vts COMMAND module_io_vts)
