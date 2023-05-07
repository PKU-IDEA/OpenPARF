#include "min_cost_flow_graph_factory.h"
#include <fstream>

namespace clock_network_planner {

decltype(auto) MinCostFlowGraphFactory::genMinCostFlowGraph(
        AssignmentGroupArray groups, FlowIntType flow_size_per_area_dem,
        CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map, const PlaceDB &db, IndexType resource_id,
        GroupToRegionDistFunctor dist_functor, IndexType site_capacity,
        RealType min_cost_max_flow_cost_scaling_factor) {
    auto        graph   = detail::makeMinCostFlowGraph();
    const auto &crs_map = db.db()->layout().clockRegionMap();
    IndexType   crs_num = crs_map.size();
    graph.cr_id2_mid_arc_idx_array().clear();
    graph.cr_id2_mid_arc_idx_array().resize(crs_num);
    graph.ck_id2mid_arc_idx_array().clear();
    graph.flow_capacity() = 0;
    graph.flow_supply()   = 0;
    // Add nodes to the left
    // Add arcs between source and left
    for (auto &group : groups) {
        graph.left_nodes().emplace_back(graph.g().addNode());
        auto &node = graph.left_nodes().back();
        graph.left_arcs().emplace_back(graph.g().addArc(graph.s(), node));
        auto &arc              = graph.left_arcs().back();
        graph.cost()[arc]      = 0;
        FlowIntType flow       = group.area_dem() * flow_size_per_area_dem;
        graph.lower_cap()[arc] = 0;
        group.flow_supply()    = flow;
        graph.upper_cap()[arc] = flow;
        graph.flow_supply() += flow;
    }
    for (IndexType cr_id = 0; cr_id < crs_map.size(); cr_id++) {
        const auto &cr = crs_map.at(cr_id);
        openparfAssert(cr.id() == cr_id);
        graph.right_nodes().emplace_back(graph.g().addNode());
        auto &node = graph.right_nodes().back();
        graph.right_arcs().emplace_back(graph.g().addArc(node, graph.t()));
        auto &arc              = graph.right_arcs().back();
        graph.cost()[arc]      = 0;
        graph.lower_cap()[arc] = 0;
        FlowIntType flow = cr_id_rsrc_id2site_map.at(cr.id(), resource_id).size() * site_capacity *
                           flow_size_per_area_dem;
        graph.upper_cap()[arc] = flow;
        graph.flow_capacity() += flow;
    }
    openparfAssert(graph.flow_supply() <= graph.flow_capacity());

    //    std::stringstream ss;
    //    ss << resource_id;
    //    std::ofstream of;
    //    of.open("mcf_cost_" + std::string(ss.str()) + ".txt");

    // Generate arcs between left (assignment groups) and right (clock regions)
    for (IndexType group_id = 0; group_id < groups.size(); group_id++) {
        auto &group = groups.at(group_id);
        for (IndexType cr_id = 0; cr_id < crs_map.size(); cr_id++) {
            const auto &cr = crs_map.at(cr_id);
            openparfAssert(cr.id() == cr_id);
            RealType dist = dist_functor(group, cr, db, cr_id_rsrc_id2site_map);
            if (dist == std::numeric_limits<RealType>::max() || dist < 0) { continue; }
            FlowIntType arc_cost   = dist * min_cost_max_flow_cost_scaling_factor;
            IndexType   mid_arc_id = graph.mid_arcs().size();
            graph.cr_id2_mid_arc_idx_array().at(cr_id).push_back(mid_arc_id);
            for (IndexType ck_id : group.ck_sets()) {
                graph.ck_id2mid_arc_idx_array()[ck_id].push_back(mid_arc_id);
            }
            graph.mid_arcs().emplace_back(
                    graph.g().addArc(graph.left_nodes()[group_id], graph.right_nodes()[cr_id]));
            graph.mid_arc_node_pairs().emplace_back(group_id, cr_id);
            auto &arc         = graph.mid_arcs().back();
            graph.cost()[arc] = arc_cost;
            //            of << group_id << " " << cr_id << ": " << dist << " " << arc_cost <<
            //            std::endl;
            graph.lower_cap()[arc] = 0;
            graph.upper_cap()[arc] = group.area_dem() * flow_size_per_area_dem;
        }
    }
    //    of.close();
    return graph;
}

MinCostFlowGraphWrapper MinCostFlowGraphFactory::genMinCostFlowGraphFromPlaceDB(
        const PlaceDB &db, CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map, ClockNetworkPlanner cnp,
        GroupArrayWrapper group_array_wrapper) {
    auto min_cost_flow_graph_wrapper = detail::makeMinCostFlowGraphWrapper();

    // NOTE: Assume a site is CLB iff. it contains the first resource id in the packing resource
    // ids array.
    min_cost_flow_graph_wrapper.packed_group_graph() = genMinCostFlowGraph(
            group_array_wrapper.packed_group_array(), cnp.flow_size_per_packed_area_dem(),
            cr_id_rsrc_id2site_map, db, cnp.packed_resource_ids()[0], getPackedGroupToCrDist,
            /*site_capacity=*/2, cnp.min_cost_max_flow_cost_scaling_factor());
    for (auto &iter : group_array_wrapper.single_ele_group_array_map()) {
        IndexType resource_id = iter.first;
        auto &    group_array = iter.second;
        auto      graph       = genMinCostFlowGraph(group_array, /*flow_size_per_area_dem=*/1,
                                         cr_id_rsrc_id2site_map, db, resource_id,
                                         getSingleEleGroupToCrDist, /*site_capacity=*/1,
                                         cnp.min_cost_max_flow_cost_scaling_factor());
        min_cost_flow_graph_wrapper.single_ele_group_graphs_map()[resource_id] = graph;
    }
    return min_cost_flow_graph_wrapper;
}
}   // namespace clock_network_planner