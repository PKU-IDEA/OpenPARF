#include "group_assignment_result_factory.h"

namespace clock_network_planner {
AssignmentResultWrapper GroupAssignmentResultFactory::storeGroupToClockRegionAssignmentSolution(
        MinCostFlowGraphWrapper mcf_graph_wrapper) {
    auto functor = [](MinCostFlowGraph graph, GroupAssignmentResultArray &result_array) {
        result_array.clear();
        result_array.resize(graph.left_nodes().size());
        for (int32_t arc_id = 0; arc_id < graph.mid_arc_node_pairs().size(); arc_id++) {
            auto &p = graph.mid_arc_node_pairs().at(arc_id);
            if (p.first == std::numeric_limits<decltype(p.first)>::max()) { continue; }
            auto &arc       = graph.mid_arcs().at(arc_id);
            auto  flow_size = graph.flow().flow(arc);
            if (flow_size > 0) { result_array.at(p.first).emplace_back(p.second, flow_size); }
        }
    };
    auto result_wrapper = detail::makeAssignmentResultWrapper();
    functor(mcf_graph_wrapper.packed_group_graph(), result_wrapper.packed_group_array());
    for (const auto &iter : mcf_graph_wrapper.single_ele_group_graphs_map()) {
        functor(iter.second, result_wrapper.single_ele_group_array_map()[iter.first]);
    }
    return result_wrapper;
}
}   // namespace clock_network_planner
