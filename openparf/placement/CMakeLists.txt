file(GLOB SRCS *.py)

install(FILES ${SRCS} DESTINATION openparf/placement)
install(DIRECTORY functor/ DESTINATION openparf/placement/functor)
