#include "max_flow_graph_factory.h"
#include "clock_info_group.h"
#include "util/message.h"

namespace clock_network_planner {

decltype(auto) MaxFlowGraphFactory::genMaxFlowGraph(ClockInfoGroupArray &clock_info_array,
                                                    FlowIntType          flow_size_per_area_dem,
                                                    const PlaceDB &      db,
                                                    CrIdRsrcId2SiteMap   cr_id_rsrc_id2site_map,
                                                    IndexType            resource_id,
                                                    IndexType            site_capacity) {
    auto        graph       = detail::makeMaxFlowGraph();
    const auto &crs_map     = db.db()->layout().clockRegionMap();
    IndexType   cr_num      = crs_map.size();
    IndexType   ck_sets_num = clock_info_array.size();
    graph.flow_supply()     = 0;
    graph.flow_capacity()   = 0;
    for (auto &group : clock_info_array) {
        graph.left_nodes().emplace_back(graph.g().addNode());
        auto &node = graph.left_nodes().back();
        graph.left_arcs().emplace_back(graph.g().addArc(graph.s(), node));
        auto &arc            = graph.left_arcs().back();
        graph.cap_map()[arc] = group.flow_supply();
        graph.flow_supply() += group.flow_supply();
    }
    for (int32_t cr_id = 0; cr_id < cr_num; cr_id++) {
        auto const &cr = crs_map.at(cr_id);
        openparfAssert(cr.id() == cr_id);
        FlowIntType cap = cr_id_rsrc_id2site_map.at(cr_id, resource_id).size() *
                          flow_size_per_area_dem * site_capacity;
        graph.right_nodes().emplace_back(graph.g().addNode());
        auto &node = graph.right_nodes().back();
        if (cap <= 0) { continue; }
        graph.right_arcs().emplace_back(graph.g().addArc(node, graph.t()));
        auto &arc            = graph.right_arcs().back();
        graph.cap_map()[arc] = cap;
        graph.flow_capacity() += cap;
    }
    openparfAssert(graph.right_nodes().size() == cr_num);
    openparfAssert(graph.flow_supply() <= graph.flow_capacity());
    for (int group_id = 0; group_id < ck_sets_num; group_id++) {
        auto &group = clock_info_array.at(group_id);
        for (int cr_id = 0; cr_id < cr_num; cr_id++) {
            auto &cr = crs_map.at(cr_id);
            openparfAssert(cr.id() == cr_id);
            graph.mid_arcs().emplace_back(graph.g().addArc(graph.left_nodes().at(group_id),
                                                           graph.right_nodes().at(cr_id)));
            auto &arc            = graph.mid_arcs().back();
            graph.cap_map()[arc] = group.flow_supply();
            graph.mid_arc_node_pairs().emplace_back(group_id, cr_id);
        }
    }
    return graph;
}


MaxFlowGraphWrapper
MaxFlowGraphFactory::genMaxFlowGraphFromPlaceDB(const PlaceDB &db, ClockNetworkPlanner cnp,
                                                ClockInfoGroupArrayWrapper clock_info_wrapper,
                                                const CrIdRsrcId2SiteMap & cr_id_rsrc_id2site_map) {
    auto max_flow_wrapper = detail::makeMaxFlowGraphWrapper();
    max_flow_wrapper.packed_group_graph() =
            genMaxFlowGraph(clock_info_wrapper.packed_clock_info_array(),
                            cnp.flow_size_per_packed_area_dem(), db, cr_id_rsrc_id2site_map,
                            /*resource_id=*/cnp.packed_resource_ids()[0], /*site_capacity=*/2);

    for (auto &iter : clock_info_wrapper.single_ele_clock_info_array_map()) {
        IndexType resource_id      = iter.first;
        auto &    clock_info_array = iter.second;
        auto      graph = genMaxFlowGraph(clock_info_array, /*flow_size_per_area_dem=*/1, db,
                                     cr_id_rsrc_id2site_map, resource_id, /*site_capacity=*/2);
        max_flow_wrapper.single_ele_group_graphs_map()[resource_id] = graph;
        openparfAssert(clock_info_array.size() == graph.left_nodes().size());
    }
    return max_flow_wrapper;
}
}   // namespace clock_network_planner