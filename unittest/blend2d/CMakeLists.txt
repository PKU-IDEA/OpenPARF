file(GLOB UNITTEST_BLEND2D_SOURCES
  *.cpp
  )

add_executable(unittest_blend2d ${UNITTEST_BLEND2D_SOURCES})
target_include_directories(unittest_blend2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../openparf
  ${Boost_HEADERS} ${PROJECT_SOURCE_DIR}/thirdparty/googletest/googletest/include)
target_link_libraries(unittest_blend2d blend2d util bookshelf Blend2D::Blend2D gtest)
install(TARGETS unittest_blend2d DESTINATION unittest/blend2d)

install(FILES unittest_blend2d.py DESTINATION unittest/blend2d)

add_test(NAME unittest_blend2d COMMAND ${CMAKE_CURRENT_BINARY_DIR}/unittest_blend2d
  ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME python_unittest_blend2d COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_blend2d.py
  ${PROJECT_BINARY_DIR})
