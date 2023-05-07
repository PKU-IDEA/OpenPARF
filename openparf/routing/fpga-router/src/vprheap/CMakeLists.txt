# -----
# @file          : CMakeLists.txt
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 23:34:08, Monday
# @brief         :
# -----
# Last Modified: April 25 2023, 01:04:41, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#
file(GLOB SOURCES *.cpp)
add_library(${PACKAGE_NAME} STATIC ${SOURCES})

target_include_directories(${PACKAGE_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(${PACKAGE_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)