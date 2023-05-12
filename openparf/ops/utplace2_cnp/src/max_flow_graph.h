#pragma once

#include "util/arg.h"
#include <cstdint>
#include <lemon/list_graph.h>   // lemon-1.3.1
#include <lemon/preflow.h>      // lemon-1.3.1
#include <memory>
#include <unordered_map>
#include <vector>

namespace clock_network_planner {

class MaxFlowGraphImpl : public std::enable_shared_from_this<MaxFlowGraphImpl> {
public:
    using IndexType             = std::int32_t;
    using FlowIntType           = std::int64_t;
    using LemonGraph            = lemon::ListDigraph;
    using Node                  = LemonGraph::Node;
    using Arc                   = LemonGraph::Arc;
    using LemonMaxFlowAlgorithm = lemon::Preflow<LemonGraph, LemonGraph::ArcMap<FlowIntType>>;
    using MidArcNodePairs       = std::vector<std::pair<IndexType, IndexType>>;

    explicit MaxFlowGraphImpl()
        : g_(), s_(g_.addNode()), t_(g_.addNode()), cap_map_(g_), flow_(g_, cap_map_, s_, t_) {}

private:
    CLASS_ARG(LemonGraph, g);                              ///< Lemon graph
    CLASS_ARG(Node, s);                                    ///< Flow source node
    CLASS_ARG(Node, t);                                    ///< Flow target node
    CLASS_ARG(LemonGraph::ArcMap<FlowIntType>, cap_map);   ///< Arc capacity map
    CLASS_ARG(std::vector<Node>, left_nodes);              ///< Clock groups
    CLASS_ARG(std::vector<Node>, right_nodes);             ///< Clock regions
    CLASS_ARG(std::vector<Arc>, left_arcs);    ///< Arcs between source/right and left/target
    CLASS_ARG(std::vector<Arc>, right_arcs);   ///< Arcs between source/right and left/target
    CLASS_ARG(std::vector<Arc>, mid_arcs);     ///< Arcs between left and right
    CLASS_ARG(LemonMaxFlowAlgorithm, flow);    ///< Lemon min-cost-max-flow object
    CLASS_ARG(MidArcNodePairs,
              mid_arc_node_pairs);               ///< Node pair (left index, right index) for midArc
    CLASS_ARG(FlowIntType, flow_supply) = 0;     ///< Flow supply, should be equal to ((total node
                                                 ///< area or count) * constant)
    CLASS_ARG(FlowIntType, flow_capacity) = 0;   ///< Flow capacity, should be equal to (total clock
                                                 ///< region area or site count)
public:
    bool run();
};

class MaxFlowGraph {
    SHARED_CLASS_ARG(MaxFlowGraph, impl);

public:
    FORWARDED_METHOD(g)
    FORWARDED_METHOD(s)
    FORWARDED_METHOD(t)
    FORWARDED_METHOD(cap_map)
    FORWARDED_METHOD(left_nodes)
    FORWARDED_METHOD(right_nodes)
    FORWARDED_METHOD(left_arcs)
    FORWARDED_METHOD(right_arcs)
    FORWARDED_METHOD(mid_arcs)
    FORWARDED_METHOD(flow)
    FORWARDED_METHOD(mid_arc_node_pairs)
    FORWARDED_METHOD(flow_supply)
    FORWARDED_METHOD(flow_capacity)
    FORWARDED_METHOD(run)
};

MAKE_SHARED_CLASS(MaxFlowGraph)

class MaxFlowGraphWrapperImpl : public std::enable_shared_from_this<MaxFlowGraphWrapperImpl> {
public:
    using IndexType   = int32_t;
    using Index2Graph = std::unordered_map<IndexType, MaxFlowGraph>;

private:
    CLASS_ARG(MaxFlowGraph, packed_group_graph);
    CLASS_ARG(Index2Graph, single_ele_group_graphs_map);

};

class MaxFlowGraphWrapper {
private:
    std::shared_ptr<MaxFlowGraphWrapperImpl> impl_;

public:
    explicit MaxFlowGraphWrapper(std::shared_ptr<MaxFlowGraphWrapperImpl> impl)
        : impl_(std::move(impl)) {}
    FORWARDED_METHOD(packed_group_graph)
    FORWARDED_METHOD(single_ele_group_graphs_map)
};

MAKE_SHARED_CLASS(MaxFlowGraphWrapper)

}   // namespace clock_network_planner
