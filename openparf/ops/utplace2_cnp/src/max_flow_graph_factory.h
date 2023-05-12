#pragma once

#include "assignment_group.h"
#include "clock_info_group.h"
#include "cr_id_rsrc_id2site_map.h"
#include "database/placedb.h"

namespace clock_network_planner {

struct MaxFlowGraphFactory {

    static decltype(auto) genMaxFlowGraph(ClockInfoGroupArray &clock_info_array,
                                          FlowIntType flow_size_per_area_dem, const PlaceDB &db,
                                          CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map,
                                          IndexType resource_id, IndexType site_capacity);

    static MaxFlowGraphWrapper
    genMaxFlowGraphFromPlaceDB(const PlaceDB &db, ClockNetworkPlanner cnp,
                               ClockInfoGroupArrayWrapper clock_info_wrapper,
                               const CrIdRsrcId2SiteMap & cr_id_rsrc_id2site_map);
};
}   // namespace clock_network_planner