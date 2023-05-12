#pragma once
#include "clock_demand_estimation.h"
#include "cr_id2ck_sink_set.h"
#include "clock_network_planner.h"

namespace clock_network_planner {

    struct ClockDemandEstimationFactory {
        static ClockDemandEstimation estimateClockDemand(
                ClockNetworkPlanner cnp, CrId2CkSinkSet cr_id2ck_sink_set);
    };
}