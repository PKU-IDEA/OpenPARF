# -----
# @file          : taskflow.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 23:20:08, Monday
# @brief         :
# -----
# Last Modified: April 25 2023, 01:01:34, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#

# The next part looks for TASKFLOW. Typically, you don't want to modify it.
# It tries to use TASKFLOW as a CMAKE subproject by looking
# for it in the 'taskflow' subdirectories or in directory given by
# jthe LEMON_SOURCE_ROOT_DIR variable.
if(NOT EXISTS TASKFLOW_SOURCE_ROOT_DIR)
    find_path(TASKFLOW_SOURCE_ROOT_DIR taskflow.hpp
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow ${THIRD_PARTY_DIR}/taskflow/taskflow
        NO_DEFAULT_PATH
        DOC "Location of TASKFLOW source as a CMAKE subproject")
endif()

if(TASKFLOW_SOURCE_ROOT_DIR)
    set(TASKFLOW_INCLUDE_DIRS ${TASKFLOW_SOURCE_ROOT_DIR}/..)
else()
    message(FATAL_ERROR "TASKFLOW source not found.")
endif()