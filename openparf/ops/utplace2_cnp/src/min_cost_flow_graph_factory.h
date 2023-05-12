#include "assignment_group.h"
#include "cr_id_rsrc_id2site_map.h"
#include "database/placedb.h"

namespace clock_network_planner {

using FlowIntType = ClockNetworkPlannerImpl::FlowIntType;
using openparf::database::PlaceDB;

struct MinCostFlowGraphFactory {
    static decltype(auto)
    genMinCostFlowGraph(AssignmentGroupArray groups, FlowIntType flow_size_per_area_dem,
                        CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map, const PlaceDB &db,
                        IndexType resource_id, GroupToRegionDistFunctor dist_functor,
                        IndexType site_capacity, RealType min_cost_max_flow_cost_scaling_factor);

    static MinCostFlowGraphWrapper
    genMinCostFlowGraphFromPlaceDB(const PlaceDB &db, CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map,
                                   ClockNetworkPlanner cnp, GroupArrayWrapper group_array_wrapper);
};

}   // namespace clock_network_planner
