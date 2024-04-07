add_library(pugixml STATIC pugixml.cpp)

target_include_directories(pugixml
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(pugixml PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

