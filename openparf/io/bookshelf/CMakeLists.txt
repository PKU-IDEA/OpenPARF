include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${FLEX_INCLUDE_DIRS}
    )

BISON_TARGET(BookshelfParser
  ${CMAKE_CURRENT_SOURCE_DIR}/bookshelf_parser.yy
  ${CMAKE_CURRENT_BINARY_DIR}/bookshelf_parser.cpp
  DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/bookshelf_parser.h
  )
FLEX_TARGET(BookshelfLexer
  ${CMAKE_CURRENT_SOURCE_DIR}/bookshelf_scanner.ll
  ${CMAKE_CURRENT_BINARY_DIR}/bookshelf_scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(BookshelfLexer BookshelfParser)

file(GLOB SOURCES
    bookshelf_driver.cpp
    )
add_library(bookshelf STATIC ${SOURCES} ${BISON_BookshelfParser_OUTPUTS} ${FLEX_BookshelfLexer_OUTPUTS})
target_include_directories(bookshelf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(bookshelf PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
#target_compile_options(bookshelf PRIVATE "-DZLIB=1")
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_definitions(bookshelf PRIVATE DEBUG_BOOKSHELFPARSER)
endif()
