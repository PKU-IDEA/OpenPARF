#pragma once
#include "group_assignment_result.h"
#include "min_cost_flow_graph.h"

namespace clock_network_planner {

struct GroupAssignmentResultFactory {

    static AssignmentResultWrapper
    storeGroupToClockRegionAssignmentSolution(MinCostFlowGraphWrapper mcf_graph_wrapper);
};
}   // namespace clock_network_planner
