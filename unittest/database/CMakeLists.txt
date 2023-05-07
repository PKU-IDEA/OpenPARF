file(GLOB UNITTEST_DATABASE_SOURCES
  *.cpp
  )

add_executable(unittest_database ${UNITTEST_DATABASE_SOURCES})
target_include_directories(unittest_database PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../openparf
  ${Boost_HEADERS} ${PROJECT_SOURCE_DIR}/thirdparty/googletest/googletest/include)
target_link_libraries(unittest_database database util bookshelf gtest)
install(TARGETS unittest_database DESTINATION unittest/database)

install(FILES unittest_database.py unittest_placedb.py
  DESTINATION unittest/database)

add_test(NAME unittest_database COMMAND ${CMAKE_CURRENT_BINARY_DIR}/unittest_database
  ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME python_unittest_database COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_database.py
  ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/benchmarks)
add_test(NAME python_unittest_placedb COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/unittest_placedb.py
  ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/benchmarks)
