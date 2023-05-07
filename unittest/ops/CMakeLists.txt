file(GLOB INSTALL_SRCS *.py)
install(FILES ${INSTALL_SRCS} DESTINATION unittest/ops)

add_test(NAME python_unittest_hpwl COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_hpwl.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_wawl COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_wawl.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_dct COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_dct.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_move_boundary COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_move_boundary.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_pin_pos COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_pin_pos.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_density_map COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_density_map.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_rudy COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unittest_rudy.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_pin_utilization COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unittest_pin_utilization.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_energy_well COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unittest_energy_well.py
  ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_fence_region_checker COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unittest_fence_region_checker.py
  ${PROJECT_BINARY_DIR})

install(DIRECTORY electric_potential DESTINATION unittest/ops)
add_test(NAME python_unittest_electric_potential COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/electric_potential/unittest_electric_potential.py
  ${PROJECT_BINARY_DIR})

install(DIRECTORY resource_area DESTINATION unittest/ops FILES_MATCHING PATTERN "*.py" PATTERN "*.pt")
add_test(NAME python_unittest_resource_area COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_area/unittest_resource_area.py
    ${PROJECT_BINARY_DIR})
add_test(NAME python_unittest_ff_ctrlsets COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_area/unittest_ff_ctrlsets.py
    ${PROJECT_BINARY_DIR})

install(DIRECTORY clock_network_planner DESTINATION unittest/ops FILES_MATCHING PATTERN "*.py" PATTERN "*.pt" PATTERN "*.json")
add_test(NAME python_unittest_clock_network_planner COMMAND ${PYTHON_EXECUTABLE}
${CMAKE_CURRENT_SOURCE_DIR}/clock_network_planner/unittest_clock_network_planner.py
    ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
