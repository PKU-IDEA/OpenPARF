/**
 * File              : min_cost_flow_graph.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 04.16.2021
 * Last Modified Date: 04.16.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "min_cost_flow_graph.h"
#include "util/message.h"

namespace clock_network_planner {

lemon::NetworkSimplex<MinCostFlowGraphImpl::LemonGraph, MinCostFlowGraphImpl::FlowIntType,
                      MinCostFlowGraphImpl::FlowIntType>::ProblemType
MinCostFlowGraphImpl::run() {
    flow_.reset();
    flow_.stSupply(s_, t_, flow_supply_);
    flow_.lowerMap(lower_cap_).upperMap(upper_cap_).costMap(cost_);
    auto        result    = flow_.run();
    FlowIntType flow_size = 0;
    for (const auto &arc : right_arcs()) { flow_size += flow_.flow(arc); }
    openparf::openparfPrint(
            openparf::kDebug,
            "Min-cost Flow: flow_size = %ld, flow_supply = %ld, flow_capacity = %ld, cost = %ld\n",
            flow_size, flow_supply_, flow_capacity_, flow_.totalCost());
    if (result == MinCostFlowGraphImpl::LemonMinCostMaxFlowAlgorithm::OPTIMAL) {
        openparfAssert(flow_size == flow_supply_);
    }
    return result;
}

int32_t MinCostFlowGraphWrapperImpl::run() {
    if(packed_group_graph_.run() != MinCostFlowGraphImpl::LemonMinCostMaxFlowAlgorithm::OPTIMAL){
        return -1;
    }
    for(auto iter : single_ele_group_graphs_map_){
        auto &graph = iter.second;
        if(graph.run() != MinCostFlowGraphImpl::LemonMinCostMaxFlowAlgorithm::OPTIMAL){
            return -1;
        }
    }
    return 0;
}

}   // namespace clock_network_planner
