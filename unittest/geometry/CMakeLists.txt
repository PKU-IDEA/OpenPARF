install(FILES unittest_geometry.py DESTINATION unittest/geometry)

add_test(NAME python_unittest_geometry COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_geometry.py
  ${PROJECT_BINARY_DIR})
