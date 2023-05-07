# -----
# @file          : CMakeLists.txt
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 22 2023, 20:44:52, Saturday
# @brief         :
# -----
# Last Modified: April 25 2023, 00:15:23, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#
add_library(pugixml STATIC pugixml.cpp)

target_include_directories(pugixml
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(pugixml PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)