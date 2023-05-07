// local dependency
#include "clock_network_planner.h"

#include "clock_demand_estimation.h"
#include "clock_demand_estimation_factory.h"
#include "clock_info_group_factory.h"
#include "clock_network_planner_factory.h"
#include "cr_id2ck_sink_set_factory.h"
#include "geometry/point.hpp"
#include "group_assignment_result_factory.h"

namespace clock_network_planner {

template<typename scalar_t>
int32_t ClockNetworkPlannerImpl::run(scalar_t *pos, scalar_t *inst_areas,
                                     ClockNetworkPlannerImpl::ClockAvailCRArray &ck_avail_crs) {
    GroupArrayWrapper       group_array_wrapper = genGroupArrayWrapper_<scalar_t>(pos, inst_areas);
    MinCostFlowGraphWrapper    mcf_graph_wrapper = genMinCostFlowGraphWrapper_(group_array_wrapper);
    ClockInfoGroupArrayWrapper clock_info_group_array_wrapper =
            ClockInfoGroupFactory::genClockInfoGroupWrapperFromGroupArrayWrapper(
                    group_array_wrapper);
    MaxFlowGraphWrapper max_flow_graph_wrapper =
            genMaxFlowGraphWrapper_(clock_info_group_array_wrapper);
    while (true) {
        if (mcf_graph_wrapper.run() < 0) {
            // can not find feasible solution
            return -1;
        }
        ArcPruningResult res = checkClockConstraintAndPruneFlowArcs_(
                group_array_wrapper, clock_info_group_array_wrapper, mcf_graph_wrapper,
                max_flow_graph_wrapper);
        if (res == ArcPruningResult::SUCCESS) {
            // Group assignment is legal
            break;
        } else if (res == ArcPruningResult::FAIL) {
            return -2;
        }
    }
    AssignmentResultWrapper group_result_wrapper =
            GroupAssignmentResultFactory::storeGroupToClockRegionAssignmentSolution(
                    mcf_graph_wrapper);
    ClockRegionAssignmentResultWrapper clock_result_wrapper =
            this->genClockRegionAssignmentResult_(group_array_wrapper, group_result_wrapper);
    this->commitResult_(clock_result_wrapper, ck_avail_crs);
    return 0;
}

template<typename scalar_t>
GroupArrayWrapper ClockNetworkPlannerImpl::genGroupArrayWrapper_(scalar_t *pos,
                                                                 scalar_t *inst_areas) {
    static_assert(sizeof(scalar_t) == 0, "Not implemented");
}

template<>
GroupArrayWrapper ClockNetworkPlannerImpl::genGroupArrayWrapper_<float>(float *pos,
                                                                        float *inst_areas) {
    return genGroupArrayWrapperFloat_(pos, inst_areas);
}

template<>
GroupArrayWrapper ClockNetworkPlannerImpl::genGroupArrayWrapper_<double>(double *pos,
                                                                         double *inst_areas) {
    return genGroupArrayWrapperDouble_(pos, inst_areas);
}



ClockNetworkPlannerImpl::IndexType
ClockNetworkPlannerImpl::selectOverflowCrToBeResolved_(ClockDemandEstimation clock_estimation) {
    int32_t   max_overflow = 0;
    IndexType target_cr_id = std::numeric_limits<IndexType>::max();
    for (IndexType cr_id = 0; cr_id < clock_estimation.ck_demand_grid().size(); cr_id++) {
        auto &  cr = clock_estimation.ck_demand_grid().at(cr_id);
        int32_t overflow =
                std::max(0, static_cast<int32_t>(cr.size()) - clock_region_capacity());
        if (overflow > max_overflow) {
            max_overflow = overflow;
            target_cr_id = cr_id;
        }
    }
    return target_cr_id;
}

bool ClockNetworkPlannerImpl::checkClockAllowedCrBoxFeasibility_(
        ClockDemandEstimation ck_estimation, ClockInfoGroupArrayWrapper clock_info_wrapper,
        MaxFlowGraphWrapper max_flow_graph_wrapper) {
    using openparf::geometry::Point;
    std::vector<MaxFlowGraph>                                graphs;
    std::vector<std::reference_wrapper<ClockInfoGroupArray>> clock_info_array;
    clock_info_array.emplace_back(clock_info_wrapper.packed_clock_info_array());
    graphs.emplace_back(max_flow_graph_wrapper.packed_group_graph());
    for (auto &iter : clock_info_wrapper.single_ele_clock_info_array_map()) {
        clock_info_array.emplace_back(iter.second);
        graphs.emplace_back(max_flow_graph_wrapper.single_ele_group_graphs_map()[iter.first]);
    }
    for (IndexType graph_id = 0; graph_id < graphs.size(); graph_id++) {
        auto &graph       = graphs.at(graph_id);
        auto &clock_infos = clock_info_array.at(graph_id);
        openparfAssert(graph.left_nodes().size() == clock_infos.get().size());
        for (IndexType mid_arc_id = 0; mid_arc_id < graph.mid_arcs().size(); mid_arc_id++) {
            auto &              np      = graph.mid_arc_node_pairs().at(mid_arc_id);
            auto &              ck_info = clock_infos.get().at(np.first);
            IndexType           cr_id   = np.second;
            IndexType           x_cr_id = cr_id / ck_estimation.ck_demand_grid().ySize();
            IndexType           y_cr_id = cr_id % ck_estimation.ck_demand_grid().ySize();
            Point<IndexType, 2> cr_p(x_cr_id, y_cr_id);
            auto &              arc   = graph.mid_arcs().at(mid_arc_id);
            bool                allow = true;
            for (IndexType ck_id : ck_info.ck_sets()) {
                const auto &box = ck_estimation.ck2cr_box_map().at(ck_id);
                if (!box.contain(cr_p)) {
                    allow = false;
                    break;
                }
            }
            graph.cap_map()[arc] = allow ? ck_info.flow_supply() : 0;
        }
        if (!graph.run()) {
            openparfPrint(kDebug, "Shrinking fails on graph %d\n", graph_id);
            return false;
        }
    }
    return true;
}

ClockNetworkPlannerImpl::ArcPruningResult
ClockNetworkPlannerImpl::checkClockConstraintAndPruneFlowArcs_(
        GroupArrayWrapper group_wrapper, ClockInfoGroupArrayWrapper clock_info_wrapper,
        MinCostFlowGraphWrapper mcf_graph_wrapper, MaxFlowGraphWrapper max_flow_graph_wrapper) {
    auto cr_id2ck_sink_set = CrId2CkSinkSetFactory::collectGroupAssignmentClockDistribution(
            group_wrapper, mcf_graph_wrapper);
    auto ck_estimation = ClockDemandEstimationFactory::estimateClockDemand(
            ClockNetworkPlanner(shared_from_this()), cr_id2ck_sink_set);
    auto target_cr_id = ClockNetworkPlannerImpl::selectOverflowCrToBeResolved_(ck_estimation);
    if (target_cr_id == std::numeric_limits<decltype(target_cr_id)>::max()) {
        return ClockNetworkPlannerImpl::ArcPruningResult::SUCCESS;
    }
    IndexSet dirty_ck_set;
    if (!shrinkClockCrBoxRegions_(target_cr_id, ck_estimation, clock_info_wrapper,
                                  max_flow_graph_wrapper, dirty_ck_set)) {
        return ClockNetworkPlannerImpl::ArcPruningResult::FAIL;
    }
    pruneClockArcs_(dirty_ck_set, mcf_graph_wrapper, ck_estimation);
    return ClockNetworkPlannerImpl::ArcPruningResult::DIRTY;
}

bool ClockNetworkPlannerImpl::shrinkClockCrBoxRegions_(
        ClockNetworkPlannerImpl::IndexType target_cr_id, ClockDemandEstimation ck_estimation,
        ClockInfoGroupArrayWrapper clock_info_wrapper, MaxFlowGraphWrapper max_flow_graph_wrapper,
        ClockNetworkPlannerImpl::IndexSet &dirty_ck_set) {
    IndexType                       x_cr_id = target_cr_id / y_crs_num_;
    IndexType                       y_cr_id = target_cr_id % y_crs_num_;
    const auto &                    ck_set  = ck_estimation.ck_demand_grid().at(target_cr_id);
    std::vector<ClockCrBoxCostPair> cand_array;
    cand_array.reserve(ck_set.size() * 4);
    for (const auto &ck_id : ck_set) {
        auto cr_box = ck_estimation.ck2cr_box_map()[ck_id];
        // left
        if (x_cr_id > cr_box.xl()) {
            cand_array.emplace_back(ck_id, cr_box);
            auto &cand = cand_array.back();
            cand.cr_box.setXH(x_cr_id - 1);
            auto tmp                         = getClockNetToClockRegionBoxDist_(ck_id, cand.cr_box);
            cand.dist                        = std::get<0>(tmp);
            cand.non_zero_avg_euclidean_dist = std::get<1>(tmp);
            cand.net_insts_num               = std::get<2>(tmp);
            cand.moved_insts_num             = std::get<3>(tmp);
        }
        // right
        if (x_cr_id < cr_box.xh()) {
            cand_array.emplace_back(ck_id, cr_box);
            auto &cand = cand_array.back();
            cand.cr_box.setXL(x_cr_id + 1);
            auto tmp                         = getClockNetToClockRegionBoxDist_(ck_id, cand.cr_box);
            cand.dist                        = std::get<0>(tmp);
            cand.non_zero_avg_euclidean_dist = std::get<1>(tmp);
            cand.net_insts_num               = std::get<2>(tmp);
            cand.moved_insts_num             = std::get<3>(tmp);
        }
        // down
        if (y_cr_id > cr_box.yl()) {
            cand_array.emplace_back(ck_id, cr_box);
            auto &cand = cand_array.back();
            cand.cr_box.setYH(y_cr_id - 1);
            auto tmp                         = getClockNetToClockRegionBoxDist_(ck_id, cand.cr_box);
            cand.dist                        = std::get<0>(tmp);
            cand.non_zero_avg_euclidean_dist = std::get<1>(tmp);
            cand.net_insts_num               = std::get<2>(tmp);
            cand.moved_insts_num             = std::get<3>(tmp);
        }
        // up
        if (y_cr_id < cr_box.yh()) {
            cand_array.emplace_back(ck_id, cr_box);
            auto &cand = cand_array.back();
            cand.cr_box.setYL(y_cr_id + 1);
            auto tmp                         = getClockNetToClockRegionBoxDist_(ck_id, cand.cr_box);
            cand.dist                        = std::get<0>(tmp);
            cand.non_zero_avg_euclidean_dist = std::get<1>(tmp);
            cand.net_insts_num               = std::get<2>(tmp);
            cand.moved_insts_num             = std::get<3>(tmp);
        }
    }
    openparfAssert(checkClockAllowedCrBoxFeasibility_(ck_estimation, clock_info_wrapper,
                                                      max_flow_graph_wrapper));
    std::sort(cand_array.begin(), cand_array.end(),
              [](auto &a, auto &b) { return a.dist < b.dist; });
    auto &  target_cr                           = ck_estimation.ck_demand_grid().at(target_cr_id);
    int32_t overflow =
            std::max(0, static_cast<int32_t>(target_cr.size()) - clock_region_capacity());
    int32_t num_pruning = std::min(overflow, max_num_clock_pruning_per_iteration());
    dirty_ck_set.clear();
    for (const auto &cand : cand_array) {
        if (dirty_ck_set.size() >= num_pruning) { break; }
        if (dirty_ck_set.find(cand.ck_id) == dirty_ck_set.end()) {
            auto old_cr_box = ck_estimation.ck2cr_box_map().at(cand.ck_id);
            ck_estimation.ck2cr_box_map().at(cand.ck_id) = cand.cr_box;
            openparfPrint(kDebug,
                          "Try Shrinking Clock Region Box of clock %d:"
                          "[%d, %d, %d, %d] -> [%d, %d, %d, %d] with (pin-moving weighted "
                          "manhattan distance/non-zero avg euclidean dist/#net insts/#moved "
                          "insts): %f/%f/%d/%d\n",
                          cand.ck_id, old_cr_box.xl(), old_cr_box.yl(), old_cr_box.xh(),
                          old_cr_box.yh(), cand.cr_box.xl(), cand.cr_box.yl(), cand.cr_box.xh(),
                          cand.cr_box.yh(), cand.dist, cand.non_zero_avg_euclidean_dist,
                          cand.net_insts_num, cand.moved_insts_num);
            if (checkClockAllowedCrBoxFeasibility_(ck_estimation, clock_info_wrapper,
                                                   max_flow_graph_wrapper)) {
                dirty_ck_set.insert(cand.ck_id);
                openparfPrint(kDebug, "Shrinking Successes.\n");
            } else {
                ck_estimation.ck2cr_box_map().at(cand.ck_id) = old_cr_box;
            }
        }
    }
    return !dirty_ck_set.empty();
}

void ClockNetworkPlannerImpl::pruneClockArcs_(ClockNetworkPlannerImpl::IndexSet &dirty_ck_set,
                                              MinCostFlowGraphWrapper            mcf_graph_wrapper,
                                              ClockDemandEstimation              ck_estimation) {
    using openparf::geometry::Point;
    std::vector<MinCostFlowGraph> graphs;
    graphs.emplace_back(mcf_graph_wrapper.packed_group_graph());
    for (auto &iter : mcf_graph_wrapper.single_ele_group_graphs_map()) {
        graphs.emplace_back(iter.second);
    }
    for (auto &graph : graphs) {
        for (auto &ck_id : dirty_ck_set) {
            if (graph.ck_id2mid_arc_idx_array().find(ck_id) ==
                graph.ck_id2mid_arc_idx_array().end()) {
                continue;
            }
            auto  cr_box            = ck_estimation.ck2cr_box_map()[ck_id];
            auto &mid_arc_idx_array = graph.ck_id2mid_arc_idx_array().at(ck_id);
            for (IndexType arc_id : mid_arc_idx_array) {
                auto &p = graph.mid_arc_node_pairs().at(arc_id);
                if (p.first == std::numeric_limits<decltype(p.first)>::max()) { continue; }
                IndexType           cr_id   = p.second;
                IndexType           x_cr_id = cr_id / ck_estimation.ck_demand_grid().ySize();
                IndexType           y_cr_id = cr_id % ck_estimation.ck_demand_grid().ySize();
                Point<IndexType, 2> cr_p(x_cr_id, y_cr_id);
                if (!cr_box.contain(cr_p)) {
                    graph.g().erase(graph.mid_arcs().at(arc_id));
                    p.first = p.second = std::numeric_limits<decltype(p.first)>::max();
                }
            }
        }
    }
}


// template instantiation
template int32_t ClockNetworkPlanner::run<double>(double *pos, double *inst_areas,
                                                  ClockAvailCRArray &ck_avail_crs);

template int32_t ClockNetworkPlanner::run<float>(float *pos, float *inst_areas,
                                                 ClockAvailCRArray &ck_avail_crs);

}   // namespace clock_network_planner