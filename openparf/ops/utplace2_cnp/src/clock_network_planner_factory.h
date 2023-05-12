#pragma once

// C++ standard library headers
#include <tuple>

// project headers
#include "database/placedb.h"

// local headers
#include "assignment_group.h"
#include "clock_network_planner.h"

namespace clock_network_planner {

using openparf::database::PlaceDB;

class ClockNetworkPlannerFactory {
public:
    using IndexBox  = ClockNetworkPlannerImpl::IndexBox;
    using IndexType = ClockNetworkPlannerImpl::IndexType;
    using RealType  = ClockNetworkPlannerImpl::RealType;

    static void
    initializeBins(PlaceDB &db, std::vector<ClockNetworkPlannerImpl::IndexBox> &bin_boxes_array,
                   LayoutMap2D<ClockNetworkPlannerImpl::BoxIndexWrapper> &bin_id_size_map);

    static void commitResult(ClockNetworkPlanner                         cnp,
                             ClockRegionAssignmentResultWrapper          region_assignment_wrapper,
                             ClockNetworkPlannerImpl::ClockAvailCRArray &ck_avail_crs, PlaceDB &db);

    static std::tuple<RealType, RealType, IndexType, IndexType>
    getClockNetToClockRegionBoxDist(IndexType ck_id, const IndexBox &cr_box,
                                    ClockNetworkPlanner cnp, PlaceDB &db);

    static ClockNetworkPlanner genClockNetworkPlannerFromPlaceDB(PlaceDB &db);
};
}   // namespace clock_network_planner
