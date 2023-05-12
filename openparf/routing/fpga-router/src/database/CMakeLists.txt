add_library(database STATIC builder.cpp module.cpp grid.cpp port.cpp pin.cpp connect.cpp)

target_include_directories(database PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})