# -----
# @file          : mcf_solver.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 23:20:08, Monday
# @brief         :
# -----
# Last Modified: April 25 2023, 00:58:55, Tuesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#

# The next part looks for MCF_SOLVER. Typically, you don't want to modify it.
# It tries to use MCF_SOLVER as a CMAKE subproject by looking
# for it in the 'mcf_solver' subdirectories or in directory given by
# jthe LEMON_SOURCE_ROOT_DIR variable.
if(NOT EXISTS MCF_SOLVER_SOURCE_ROOT_DIR)
    find_path(MCF_SOLVER_SOURCE_ROOT_DIR CMakeLists.txt
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/mcf_solver ${THIRD_PARTY_DIR}/mcf_solver
        NO_DEFAULT_PATH
        DOC "Location of MCF_SOLVER source as a CMAKE subproject")
endif()

if(MCF_SOLVER_SOURCE_ROOT_DIR)
    add_subdirectory(${MCF_SOLVER_SOURCE_ROOT_DIR})
    set(MCF_SOLVER_INCLUDE_DIRS ${MCF_SOLVER_SOURCE_ROOT_DIR})
    set(MCF_SOLVER_LIBRARIES mcf_solver)
else()
    message(FATAL_ERROR "MCF_SOLVER source not found.")
endif()