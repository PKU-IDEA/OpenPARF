#pragma once

#include "assignment_group.h"
#include "clock_demand_estimation.h"
#include "clock_info_group.h"
#include "clock_region_assignment_result.h"
#include "container/index_wrapper.hpp"
#include "cr_id2ck_sink_set.h"
#include "database/layout_map.hpp"
#include "geometry/box.hpp"
#include "group_assignment_result.h"
#include "max_flow_graph.h"
#include "min_cost_flow_graph.h"
#include <memory>
#include <utility>
#include <vector>

namespace clock_network_planner {

using openparf::container::IndexWrapper;
using openparf::database::LayoutMap2D;
using openparf::geometry::Box;

class ClockNetworkPlanner;

class ClockNetworkPlannerFactory;

class ClockNetworkPlannerImpl : public std::enable_shared_from_this<ClockNetworkPlannerImpl> {
public:
    using IndexType         = int32_t;
    using RealType          = float;
    using IndexSet          = std::unordered_set<IndexType>;
    using FlowIntType       = std::int64_t;
    using IndexBox          = Box<IndexType>;
    using BoxIndexWrapper   = IndexWrapper<IndexBox, IndexType>;
    using ClockAvailCRArray = std::vector<std::vector<IndexType>>;


    /*! Class for allowed clock region box shrinking
     */
    struct ClockCrBoxCostPair {
        ClockCrBoxCostPair(IndexType _ck_id, const IndexBox &_cr_box)
            : ck_id(_ck_id), cr_box(_cr_box) {}

        IndexType ck_id;
        IndexBox  cr_box;
        RealType  dist{};
        RealType  non_zero_avg_euclidean_dist{};
        IndexType net_insts_num{};
        IndexType moved_insts_num{};
    };

    friend class ClockNetworkPlanner;

    friend class ClockNetworkPlannerFactory;

    template<typename scalar_t>
    int32_t run(scalar_t *pos, scalar_t *inst_areas, ClockAvailCRArray &ck_avail_crs);


    /**
     * Enum for arc pruning result
     */
    enum class ArcPruningResult
    {
        SUCCESS,   ///< No edge pruning is performed
        DIRTY,     ///<  Some edge is pruned
        FAIL       ///<  No edge can be pruned and the solution is not legal
    };

private:
    IndexType selectOverflowCrToBeResolved_(ClockDemandEstimation clock_estimation);

    ArcPruningResult checkClockConstraintAndPruneFlowArcs_(
            GroupArrayWrapper group_wrapper, ClockInfoGroupArrayWrapper clock_info_wrapper,
            MinCostFlowGraphWrapper mcf_graph_wrapper, MaxFlowGraphWrapper max_flow_graph_wrapper);

    bool shrinkClockCrBoxRegions_(IndexType target_cr_id, ClockDemandEstimation ck_estimation,
                                  ClockInfoGroupArrayWrapper clock_info_wrapper,
                                  MaxFlowGraphWrapper        max_flow_graph_wrapper,
                                  IndexSet &                 dirty_ck_set);

    static bool checkClockAllowedCrBoxFeasibility_(ClockDemandEstimation      ck_estimation,
                                                   ClockInfoGroupArrayWrapper clock_info_wrapper,
                                                   MaxFlowGraphWrapper max_flow_graph_wrapper);

    void pruneClockArcs_(IndexSet &dirty_ck_set, MinCostFlowGraphWrapper mcf_graph_wrapper,
                         ClockDemandEstimation ck_estimation);

private:
    CLASS_ARG(IndexType,
              x_cnp_bin_size);   ///< The bin size on the x-coordinate for aggregating LUTs & FFs.
    CLASS_ARG(IndexType,
              y_cnp_bin_size);   ///< The bin size on the y-coordinate for aggregating LUTs & FFs.
    CLASS_ARG(std::vector<IndexBox>,
              bin_boxes_array);   ///< Array of bin boxes for aggregating LUTs & FFs.
    CLASS_ARG(LayoutMap2D<BoxIndexWrapper>, bin_id_size_map);   ///< The bin index for each size
    CLASS_ARG(IndexType, clock_region_capacity);                ///< Clock region capacity.
    CLASS_ARG(IndexType,
              max_num_clock_pruning_per_iteration);   ///< Maximum number of clock pruning per
                                                      ///< min-cost flow iteration
    CLASS_ARG(bool, enable_packing);   ///< Whether enable packing for nearby LUTs & FFs

    CLASS_ARG(IndexType, x_crs_num);   ///< the number of columns for clock region grid system.
    CLASS_ARG(IndexType, y_crs_num);   ///< the number of rows for clock region grid system.
    CLASS_ARG(IndexType, crs_num);     ///< The number of clock regions.
    CLASS_ARG(IndexType, insts_num);   ///< The number of instances.
    CLASS_ARG(IndexType, cks_num);     ///< The number of clocks.
    CLASS_ARG(std::vector<RealType>, inst_areas);   ///< Instance estimated areas
    CLASS_ARG(std::vector<RealType>, inst_x_pos);   ///< Instance x coordinates
    CLASS_ARG(std::vector<RealType>, inst_y_pos);   ///< Instance y coordinates

    CLASS_ARG(std::vector<IndexType>, packed_resource_ids);
    CLASS_ARG(std::vector<IndexType>, single_ele_group_resource_ids);
    CLASS_ARG(FlowIntType, flow_size_per_packed_area_dem);
    CLASS_ARG(RealType, min_cost_max_flow_cost_scaling_factor);

    template<typename scalar_t>
    GroupArrayWrapper genGroupArrayWrapper_(scalar_t *pos, scalar_t *inst_areas);

    std::function<GroupArrayWrapper(float *, float *)>   genGroupArrayWrapperFloat_;
    std::function<GroupArrayWrapper(double *, double *)> genGroupArrayWrapperDouble_;

    std::function<MinCostFlowGraphWrapper(GroupArrayWrapper)> genMinCostFlowGraphWrapper_;


    std::function<MaxFlowGraphWrapper(ClockInfoGroupArrayWrapper)> genMaxFlowGraphWrapper_;

    std::function<std::tuple<RealType, RealType, IndexType, IndexType>(IndexType, const IndexBox &)>
            getClockNetToClockRegionBoxDist_;

    std::function<ClockRegionAssignmentResultWrapper(GroupArrayWrapper, AssignmentResultWrapper)>
            genClockRegionAssignmentResult_;

    std::function<void(ClockRegionAssignmentResultWrapper, ClockAvailCRArray &)> commitResult_;
};

class ClockNetworkPlanner {
private:
    std::shared_ptr<ClockNetworkPlannerImpl> impl_;

public:
    explicit ClockNetworkPlanner(std::shared_ptr<ClockNetworkPlannerImpl> impl)
        : impl_(std::move(impl)) {}

    using ClockAvailCRArray = ClockNetworkPlannerImpl::ClockAvailCRArray;

    FORWARDED_METHOD(insts_num)
    FORWARDED_METHOD(crs_num)
    FORWARDED_METHOD(cks_num)
    FORWARDED_METHOD(bin_id_size_map)
    FORWARDED_METHOD(clock_region_capacity)
    FORWARDED_METHOD(max_num_clock_pruning_per_iteration)
    FORWARDED_METHOD(enable_packing)
    FORWARDED_METHOD(packed_resource_ids)
    FORWARDED_METHOD(single_ele_group_resource_ids)
    FORWARDED_METHOD(x_crs_num)
    FORWARDED_METHOD(y_crs_num)
    FORWARDED_METHOD(inst_areas)
    FORWARDED_METHOD(inst_x_pos)
    FORWARDED_METHOD(inst_y_pos)
    FORWARDED_METHOD(flow_size_per_packed_area_dem)
    FORWARDED_METHOD(min_cost_max_flow_cost_scaling_factor)

    template<typename scalar_t>
    int32_t run(scalar_t *pos, scalar_t *inst_areas, ClockAvailCRArray &ck_avail_crs) {
        return impl_->run<scalar_t>(pos, inst_areas, ck_avail_crs);
    }
};

MAKE_SHARED_CLASS(ClockNetworkPlanner)

}   // namespace clock_network_planner
