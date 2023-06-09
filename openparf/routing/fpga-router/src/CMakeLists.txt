# -----
# @file          : CMakeLists.txt
# @project       : OpenPARF
# @author        : Jing Mai <jingmai@pku.edu.cn>
# @created date  : April 22 2023, 20:49:01, Saturday
# @brief         :
# -----
# Last Modified: May 05 2023, 14:28:49, Friday
# Modified By: Jing Mai <jingmai@pku.edu.cn>
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
# -----
# This header is generated by VSCode extension psi-header.
#

set(FPGAROUTER_LIB_TARGET fpgaroute)

set(PATH_SRC ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE SOURCE_FILE ${PATH_SRC}/*.cpp ${PATH_SRC}/*.c)

add_library(${FPGAROUTER_LIB_TARGET} STATIC ${SOURCE_FILE})

target_include_directories(${FPGAROUTER_LIB_TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ZLIB_INCLUDE_DIRS}
  ${GUROBI_INCLUDE_DIRS}
  ${LEMON_INCLUDE_DIRS}
  ${PUGIXML_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CLIPP_INCLUDE_DIRS}
  ${MCF_SOLVER_INCLUDE_DIRS}
  ${TASKFLOW_INCLUDE_DIRS}
)

target_link_libraries(${FPGAROUTER_LIB_TARGET} PRIVATE
  ${LAPACK_LIBRARIES}
  ${ZLIB_LIBRARIES}
  OpenMP::OpenMP_CXX
  Threads::Threads
  ${LEMON_LIBRARIES}
  ${PUGIXML_LIBRARIES}
  ${MCF_SOLVER_LIBRARIES}
  ${Boost_LIBRARY_DIRS}
  ${GUROBI_LIBRARIES}
)

set_target_properties(${FPGAROUTER_LIB_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(FPGAROUTER_LIB ${FPGAROUTER_LIB_TARGET} CACHE STRING "fpgaroute library name")