configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/configure.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/configure.py DESTINATION openparf)
install(DIRECTORY py_utils DESTINATION openparf)
install(DIRECTORY py_config DESTINATION openparf)

add_subdirectory(io)
add_subdirectory(database)
add_subdirectory(util)
add_subdirectory(pybind)
add_subdirectory(custom_data)
add_subdirectory(ops)
add_subdirectory(placement)
if (ENABLE_ROUTER)
  add_subdirectory(routing)
endif()

#add_executable(OpenPARF main.cpp)
#target_include_directories(OpenPARF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_HEADERS})
#target_link_libraries(OpenPARF PUBLIC database util bookshelf)
#set_target_properties(OpenPARF PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/py_utils
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/py_config
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(GLOB SRCS *.py *.json)
install(FILES ${SRCS} DESTINATION openparf)
