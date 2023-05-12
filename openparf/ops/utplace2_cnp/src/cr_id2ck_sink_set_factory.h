#pragma once

#include "cr_id2ck_sink_set.h"
#include "assignment_group.h"
#include "min_cost_flow_graph.h"

namespace clock_network_planner {
    struct CrId2CkSinkSetFactory {
        static CrId2CkSinkSet collectGroupAssignmentClockDistribution(
                GroupArrayWrapper group_array_wrapper,
                MinCostFlowGraphWrapper mcf_graph_wrapper
        );
    };
}