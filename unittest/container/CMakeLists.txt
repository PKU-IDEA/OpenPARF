file(GLOB UNITTEST_CONTAINER_SOURCES
        *.cpp
  )

add_executable(unittest_container ${UNITTEST_CONTAINER_SOURCES})
target_include_directories(unittest_container PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../openparf
  ${PROJECT_SOURCE_DIR}/thirdparty/googletest/googletest/include)
target_link_libraries(unittest_container util gtest)
install(TARGETS unittest_container DESTINATION unittest/container)

add_test(NAME unittest_container COMMAND ${CMAKE_CURRENT_BINARY_DIR}/unittest_container
  ${CMAKE_CURRENT_SOURCE_DIR})
