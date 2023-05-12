include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${FLEX_INCLUDE_DIRS}
    )

BISON_TARGET(VerilogParser
             VerilogParser.yy
             ${CMAKE_CURRENT_BINARY_DIR}/VerilogParser.cc
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/VerilogParser.h
             )
FLEX_TARGET(VerilogLexer
            VerilogScanner.ll
            ${CMAKE_CURRENT_BINARY_DIR}/VerilogScanner.cc)
ADD_FLEX_BISON_DEPENDENCY(VerilogLexer VerilogParser)

file(GLOB SOURCES
    VerilogDataBase.cc VerilogDriver.cc
    )
add_library(verilogparser STATIC ${SOURCES} ${BISON_VerilogParser_OUTPUTS} ${FLEX_VerilogLexer_OUTPUTS})
set_target_properties(verilogparser PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(verilogparser PRIVATE DEBUG_VERILOGPARSER)
endif()