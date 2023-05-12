#include "clock_region_assignment_result_factory.h"
#include "group_assignment_result.h"
#include "util/message.h"
#include <lemon/list_graph.h>        // lemon-1.3.1
#include <lemon/network_simplex.h>   // lemon-1.3.1

namespace clock_network_planner {

using CrToInstIdArray = ClockRegionAssignmentResultWrapperImpl::CrToInstIdArray;
using IndexType       = std::int32_t;
using IndexVector     = std::vector<IndexType>;
using FlowIntType     = std::int64_t;
using RealType        = float;
using openparf::database::ClockRegion;
using openparf::geometry::Point;

RealType ClockRegionAssignmentResultFactory::getInstToCrDist(ClockNetworkPlanner cnp,
                                                                IndexType           inst_id,
                                                                const ClockRegion & cr,
                                                                const PlaceDB &     db) {
    // Get clock region to group weighted distance
    auto geometry_bbox = cr.geometry_bbox();
    auto distXY        = geometry_bbox.manhDist(
            Point<RealType, 2>(cnp.inst_x_pos()[inst_id], cnp.inst_y_pos()[inst_id]));
    RealType x_wirelength_weight_ = db.wirelengthWeights().first;
    RealType y_wirelength_weight_ = db.wirelengthWeights().second;
    RealType dist =
            1.0 * x_wirelength_weight_ * distXY.x() + 1.0 * y_wirelength_weight_ * distXY.y();
    return db.instPins().at(inst_id).size() * dist;
}

static void splitAssignmentGroupInToClockRegions(
        AssignmentGroup &group, std::vector<GroupToRegionAssignmentResult> &group_assignment,
        CrToInstIdArray &cr_array, ClockNetworkPlanner cnp, PlaceDB &db) {
    using LemonGraph                   = lemon::ListDigraph;
    using Node                         = LemonGraph::Node;
    using Arc                          = LemonGraph::Arc;
    using LemonMinCostMaxFlowAlgorithm = lemon::NetworkSimplex<LemonGraph, FlowIntType>;
    using MidArcNodePairs              = std::vector<std::pair<IndexType, IndexType>>;

    LemonGraph                      g;
    LemonGraph::ArcMap<FlowIntType> lower_cap(g);
    LemonGraph::ArcMap<FlowIntType> upper_cap(g);
    LemonGraph::ArcMap<FlowIntType> cost(g);
    std::vector<Node>               left_nodes, right_nodes;
    std::vector<Arc>                left_arcs, right_arcs, mid_arcs;
    MidArcNodePairs                 mid_arc_node_pairs;

    FlowIntType flow_supply   = 0;
    FlowIntType flow_capacity = 0;
    Node        s             = g.addNode();
    Node        t             = g.addNode();

    for (IndexType inst_id : group.inst_ids()) {
        left_nodes.emplace_back(g.addNode());
        auto &node = left_nodes.back();
        left_arcs.emplace_back(g.addArc(s, node));
        auto &arc       = left_arcs.back();
        lower_cap[arc]  = 0;
        cost[arc]       = 0;
        FlowIntType cap = cnp.inst_areas().at(inst_id) * cnp.flow_size_per_packed_area_dem();
        upper_cap[arc]  = cap;
        flow_supply += cap;
    }

    for (auto &p : group_assignment) {
        right_nodes.emplace_back(g.addNode());
        auto &node = right_nodes.back();
        right_arcs.emplace_back(g.addArc(node, t));
        auto &arc       = right_arcs.back();
        lower_cap[arc]  = 0;
        cost[arc]       = 0;
        FlowIntType cap = std::ceil(1.0 * p.flow_size() / group.flow_supply() * flow_supply);
        flow_capacity += cap;
        upper_cap[arc] = cap;
    }
    openparfAssert(flow_supply <= flow_capacity);

    for (int32_t i = 0; i < group.inst_ids().size(); i++) {
        IndexType inst_id = group.inst_ids().at(i);
        for (int32_t j = 0; j < group_assignment.size(); j++) {
            auto &p  = group_assignment[j];
            auto &cr = db.db()->layout().clockRegionMap().at(p.cr_id());
            mid_arcs.emplace_back(g.addArc(left_nodes.at(i), right_nodes.at(j)));
            auto &arc = mid_arcs.back();
            mid_arc_node_pairs.emplace_back(i, j);
            cost[arc] = ClockRegionAssignmentResultFactory::getInstToCrDist(cnp, inst_id, cr, db) * cnp.min_cost_max_flow_cost_scaling_factor();
            lower_cap[arc] = 0;
            upper_cap[arc] = cnp.inst_areas().at(inst_id) * cnp.flow_size_per_packed_area_dem();
        }
    }
    LemonMinCostMaxFlowAlgorithm flow(g);
    flow.stSupply(s, t, flow_supply);
    flow.lowerMap(lower_cap).upperMap(upper_cap).costMap(cost);
    auto result = flow.run();
    openparfAssert(result == LemonMinCostMaxFlowAlgorithm::OPTIMAL);

    struct CrIdAndFlowSizePair {
        IndexType   cr_id;
        FlowIntType flow_size = 0;
    };
    std::vector<CrIdAndFlowSizePair> left_node_id2CrArray(group.inst_ids().size());
    for (IndexType arc_id = 0; arc_id < mid_arc_node_pairs.size(); arc_id++) {
        auto &      p                        = mid_arc_node_pairs.at(arc_id);
        auto &      arc                      = mid_arcs.at(arc_id);
        FlowIntType flow_size                = flow.flow(arc);
        auto &      cr_id_and_flow_size_pair = left_node_id2CrArray.at(p.first);
        if (flow_size > cr_id_and_flow_size_pair.flow_size) {
            cr_id_and_flow_size_pair.flow_size = flow_size;
            cr_id_and_flow_size_pair.cr_id     = group_assignment[p.second].cr_id();
        }
    }
    for (IndexType left_node_id = 0; left_node_id < left_node_id2CrArray.size(); left_node_id++) {
        IndexType inst_id = group.inst_ids().at(left_node_id);
        IndexType cr_id   = left_node_id2CrArray.at(left_node_id).cr_id;
        cr_array.at(cr_id).emplace_back(inst_id);
    }
}

static void nodeToClockRegionAssignment(AssignmentGroupArray &      group_array,
                                        GroupAssignmentResultArray &group_assignment_array,
                                        CrToInstIdArray &cr_array, ClockNetworkPlanner cnp,
                                        PlaceDB &db) {
    cr_array.clear();
    cr_array.resize(cnp.crs_num());
    for (int32_t group_id = 0; group_id < group_array.size(); group_id++) {
        auto &cr_id_and_flow_size_pair_array = group_assignment_array.at(group_id);
        auto &group                          = group_array.at(group_id);
        openparfAssert(!cr_id_and_flow_size_pair_array.empty());
        if (cr_id_and_flow_size_pair_array.size() == 1) {
            // All nodes in this groups are assigned to one clock region
            int32_t cr_id         = cr_id_and_flow_size_pair_array.at(0).cr_id();
            auto &  cr_node_array = cr_array.at(cr_id);
            cr_node_array.insert(cr_node_array.end(), group.inst_ids().begin(),
                                 group.inst_ids().end());
        } else {
            splitAssignmentGroupInToClockRegions(group, cr_id_and_flow_size_pair_array, cr_array,
                                                 cnp, db);
        }
    }
}

ClockRegionAssignmentResultWrapper
ClockRegionAssignmentResultFactory::genClockRegionAssignmentResult(
        GroupArrayWrapper group_wrapper, AssignmentResultWrapper group_result_wrapper,
        ClockNetworkPlanner cnp, PlaceDB &db) {
    auto region_result_wrapper = detail::makeClockRegionAssignmentResultWrapper();
    nodeToClockRegionAssignment(group_wrapper.packed_group_array(),
                                group_result_wrapper.packed_group_array(),
                                region_result_wrapper.packed_inst_array(), cnp, db);
    for (auto &iter : group_result_wrapper.single_ele_group_array_map()) {
        nodeToClockRegionAssignment(
                group_wrapper.single_ele_group_array_map()[iter.first], iter.second,
                region_result_wrapper.single_ele_inst_map()[iter.first], cnp, db);
    }
    auto &cr_to_all_inst_array = region_result_wrapper.cr_to_all_inst_array();
    cr_to_all_inst_array.clear();
    cr_to_all_inst_array.resize(cnp.crs_num());
    for (IndexType cr_id = 0; cr_id < cnp.crs_num(); cr_id++) {
        cr_to_all_inst_array[cr_id].insert(cr_to_all_inst_array[cr_id].end(),
                                           region_result_wrapper.packed_inst_array()[cr_id].begin(),
                                           region_result_wrapper.packed_inst_array()[cr_id].end());
        for (auto &iter : region_result_wrapper.single_ele_inst_map()) {
            cr_to_all_inst_array[cr_id].insert(cr_to_all_inst_array[cr_id].end(),
                                               iter.second.at(cr_id).begin(),
                                               iter.second.at(cr_id).end());
        }
    }
    return region_result_wrapper;
}
}   // namespace clock_network_planner