# Clock network planner algorithm for UTPLace2.0

set(SOURCE_FILES
        clock_network_planner.cpp
        assignment_group.cpp
        group_assignment_result.cpp
        min_cost_flow_graph.cpp
        max_flow_graph.cpp
        clock_info_group.cpp
        clock_info_group_factory.cpp
        cr_id_rsrc_id2site_map.cpp
        cr_id_rsrc_id2site_map_factory.cpp
        min_cost_flow_graph_factory.cpp
        max_flow_graph_factory.cpp
        assignment_group_factory.cpp
        cr_id2ck_sink_set.cpp
        cr_id2ck_sink_set_factory.cpp
        clock_network_planner_factory.cpp
        clock_demand_estimation.cpp
        clock_demand_estimation_factory.cpp
        clock_region_assignment_result.cpp
        clock_region_assignment_result_factory.cpp
        group_assignment_result_factory.cpp)

set(INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/openparf
        ${PROJECT_BINARY_DIR}/thirdparty/lemon
        ${PROJECT_SOURCE_DIR}/thirdparty/lemon
        ${Boost_INCLUDE_DIRS})

set(LINK_LIBS database bookshelf util lemon)

add_library(${TARGET_NAME} STATIC ${SOURCE_FILES})
set_property(TARGET ${TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC ${LINK_LIBS})