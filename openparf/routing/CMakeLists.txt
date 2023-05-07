# -----
# @file          : CMakeLists.txt
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : March 30 2023, 21:43:59, Thursday
# @brief         :
# -----
# Last Modified: May 05 2023, 17:21:37, Friday
# Modified By: Jing Mai <jingmai@pku.edu.cn>
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#
set(OP_NAME router)
set(SUB_MODULE_DIR fpga-router)

add_subdirectory(${SUB_MODULE_DIR})

file(GLOB CPP_SOURCES src/*.cpp)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/${SUB_MODULE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(LINK_LIBS
  ${FPGAROUTER_LIB}
)

add_pytorch_extension(${TARGET_NAME}_cpp ${CPP_SOURCES}
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS ${TARGET_NAME}_cpp DESTINATION openparf/routing)
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION openparf/routing)