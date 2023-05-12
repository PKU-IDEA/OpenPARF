set(OP_NAME utplace2_cnp)

set(TARGET_NAME "${OP_NAME}")
set(TORCH_TARGET_NAME "${OP_NAME}_torch")

add_subdirectory(src)

set(CPP_SOURCES utplace2_cnp_torch.cpp)
set(INCLUDE_DIRS)
set(LINK_LIBS ${TARGET_NAME})

add_pytorch_extension(${TORCH_TARGET_NAME} ${CPP_SOURCES}
        EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
        EXTRA_LINK_LIBRARIES ${LINK_LIBS})
install(TARGETS ${TORCH_TARGET_NAME} DESTINATION "openparf/ops/${OP_NAME}")

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION "openparf/ops/${OP_NAME}")

