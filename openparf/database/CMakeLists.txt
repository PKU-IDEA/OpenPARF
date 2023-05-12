file(GLOB SRCS *.cpp)

add_library(database STATIC ${SRCS})
target_include_directories(database PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} )
target_link_libraries(database PUBLIC pugixml::pugixml verilogparser yaml-cpp)
set_target_properties(database PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
