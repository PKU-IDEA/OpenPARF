file(GLOB CPP_SOURCES *.cpp)

add_library(util STATIC ${CPP_SOURCES})
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/.. ${Boost_INCLUDE_DIRS})
#target_link_libraries(util Boost::headers)
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
