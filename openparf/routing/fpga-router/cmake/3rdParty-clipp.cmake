# -----
# @file          : clipp.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 23:20:08, Monday
# @brief         :
# -----
# Last Modified: April 25 2023, 00:54:19, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#

# The next part looks for CLIPP. Typically, you don't want to modify it.
# It tries to use CLIPP as a CMAKE subproject by looking
# for it in the 'clipp' subdirectories or in directory given by
# jthe LEMON_SOURCE_ROOT_DIR variable.
if(NOT EXISTS CLIPP_SOURCE_ROOT_DIR)
    find_path(CLIPP_SOURCE_ROOT_DIR clipp.h
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/clipp/clipp ${THIRD_PARTY_DIR}/clipp/clipp
        NO_DEFAULT_PATH
        DOC "Location of CLIPP source as a CMAKE subproject")
endif()

if(CLIPP_SOURCE_ROOT_DIR)
    set(CLIPP_INCLUDE_DIRS ${CLIPP_SOURCE_ROOT_DIR}/..)
else()
    message(FATAL_ERROR "CLIPP source not found.")
endif()