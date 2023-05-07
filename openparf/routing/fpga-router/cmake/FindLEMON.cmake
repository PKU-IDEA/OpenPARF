# -----
# @file          : FindLEMON.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 20:42:42, Monday
# @brief         :
# -----
# Last Modified: April 24 2023, 20:44:16, Monday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
# Modified from http://lemon.cs.elte.hu/trac/lemon/browser/lemon-project-template/cmake/FindLEMON.cmake

SET(LEMON_ROOT_DIR "" CACHE PATH "LEMON root directory")

FIND_PATH(LEMON_INCLUDE_DIR
    lemon/core.h
    HINTS ${LEMON_ROOT_DIR}/include
    "C:/Program Files/LEMON/include"
)
FIND_LIBRARY(LEMON_LIBRARY
    NAMES lemon emon
    HINTS ${LEMON_ROOT_DIR}/lib
    "C:/Program Files/LEMON/lib"
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LEMON DEFAULT_MSG LEMON_LIBRARY LEMON_INCLUDE_DIR)

IF(LEMON_FOUND)
    SET(LEMON_INCLUDE_DIRS ${LEMON_INCLUDE_DIR})
    SET(LEMON_LIBRARIES ${LEMON_LIBRARY})
ENDIF(LEMON_FOUND)

MARK_AS_ADVANCED(LEMON_LIBRARY LEMON_INCLUDE_DIR)