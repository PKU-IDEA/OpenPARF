# -----
# @file          : pugixml.cmake
# @project       : OpenPARF
# @author        : Jing Mai (jingmai@pku.edu.cn)
# @created date  : April 24 2023, 21:30:10, Monday
# @brief         :
# -----
# Last Modified: April 26 2023, 21:10:44, Wednesday
# Modified By: Jing Mai (jingmai@pku.edu.cn)
# -----
# @history :
# ====================================================================================
# Date         	By     	(version)	Comments
# -------------	-------	---------	--------------------------------------------------
# ====================================================================================
# Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
#

# The next part looks for PUGIXML. Typically, you don't want to modify it.
# It tries to use PUGIXML as a CMAKE subproject by looking
# for it in the 'pugixml' subdirectories or in directory given by
# jthe LEMON_SOURCE_ROOT_DIR variable.
if(NOT EXISTS PUGIXML_SOURCE_ROOT_DIR)
    find_path(PUGIXML_SOURCE_ROOT_DIR CMakeLists.txt
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/pugixml/pugixml ${THIRD_PARTY_DIR}/pugixml/pugixml
        NO_DEFAULT_PATH
        DOC "Location of PUGIXML source as a CMAKE subproject")
endif()

if(PUGIXML_SOURCE_ROOT_DIR)
    add_subdirectory(${PUGIXML_SOURCE_ROOT_DIR})
    set(PUGIXML_INCLUDE_DIRS ${PUGIXML_SOURCE_ROOT_DIR}/..)
    set(PUGIXML_LIBRARIES pugixml)
else()
    message(FATAL_ERROR "PUGIXML source not found.")
endif()