# -----
# @file          : lemon.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 21:28:17, Monday
# @brief         :
# -----
# Last Modified: April 25 2023, 00:37:31, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#

# Modified from http://lemon.cs.elte.hu/trac/lemon/browser/lemon-project-template/CMakeLists.txt
#
# The next part looks for LEMON. Typically, you don't want to modify it.
#
# First, it tries to use LEMON as a CMAKE subproject by looking
# for it in the 'lemon' subdirectories or in directory given by
# jthe LEMON_SOURCE_ROOT_DIR variable.
#
# If LEMON isn't there, then CMAKE will try to find an installed
# version of LEMON. If it is installed at some non-standard place,
# then you must tell its location in the LEMON_ROOT_DIR CMAKE config
# variable. (Do not hard code it into your config! Others may keep
# LEMON at different places.)
if(NOT EXISTS LEMON_SOURCE_ROOT_DIR)
    FIND_PATH(LEMON_SOURCE_ROOT_DIR CMakeLists.txt
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lemon ${THIRD_PARTY_DIR}/lemon
        NO_DEFAULT_PATH
        DOC "Location of LEMON source as a CMAKE subproject")
endif()

if(LEMON_SOURCE_ROOT_DIR)
    add_subdirectory(${LEMON_SOURCE_ROOT_DIR})
    get_directory_property(LEMON_BINART_ROOT_DIR DIRECTORY ${LEMON_SOURCE_ROOT_DIR} BINARY_DIR)
    SET_TARGET_PROPERTIES(lemon PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    set(LEMON_INCLUDE_DIRS
        ${LEMON_SOURCE_ROOT_DIR}
        ${LEMON_BINART_ROOT_DIR}
    )
    set(LEMON_LIBRARIES lemon)
    unset(LEMON_ROOT_DIR CACHE)
    unset(LEMON_DIR CACHE)
    unset(LEMON_INCLUDE_DIR CACHE)
    unset(LEMON_LIBRARY CACHE)
else()
    find_package(LEMON QUIET NO_MODULE)
    find_package(LEMON REQUIRED)
endif()