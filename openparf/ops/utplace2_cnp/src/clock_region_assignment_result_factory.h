/**
 * File              : clock_region_assignment_result_factory.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 04.20.2021
 * Last Modified Date: 04.20.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#pragma once
#include "assignment_group.h"
#include "clock_network_planner.h"
#include "clock_region_assignment_result.h"
#include "database/clock_region.h"
#include "database/placedb.h"
#include "group_assignment_result.h"

namespace clock_network_planner {
using openparf::database::ClockRegion;
using openparf::database::PlaceDB;

struct ClockRegionAssignmentResultFactory {

    using IndexType   = std::int32_t;
    using FlowIntType = std::int64_t;
    using RealType    = float;

    static RealType getInstToCrDist(ClockNetworkPlanner cnp, IndexType inst_id,
                                    const ClockRegion &cr, const PlaceDB &db);
    static ClockRegionAssignmentResultWrapper
    genClockRegionAssignmentResult(GroupArrayWrapper       group_wrapper,
                                   AssignmentResultWrapper group_result_wrapper,
                                   ClockNetworkPlanner cnp, PlaceDB &db);
};
}   // namespace clock_network_planner
