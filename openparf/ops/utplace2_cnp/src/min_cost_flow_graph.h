#pragma once

#include "util/arg.h"
#include <cstdint>
#include <lemon/list_graph.h>        // lemon-1.3.1
#include <lemon/network_simplex.h>   // lemon-1.3.1
#include <memory>
#include <unordered_map>
#include <vector>

namespace clock_network_planner {

class MinCostFlowGraphImpl : public std::enable_shared_from_this<MinCostFlowGraphImpl> {
public:
    using IndexType                    = std::int32_t;
    using FlowIntType                  = std::int64_t;
    using IndexVector                  = std::vector<IndexType>;
    using LemonGraph                   = lemon::ListDigraph;
    using Node                         = LemonGraph::Node;
    using Arc                          = LemonGraph::Arc;
    using LemonMinCostMaxFlowAlgorithm = lemon::NetworkSimplex<LemonGraph, FlowIntType>;
    using CrIdToMidArcIdxArray         = std::vector<IndexVector>;
    using CkNetIdToMidArcIdxArray      = std::unordered_map<IndexType, IndexVector>;
    using MidArcNodePairs              = std::vector<std::pair<IndexType, IndexType>>;

    explicit MinCostFlowGraphImpl()
        : g_(), s_(g_.addNode()), t_(g_.addNode()), lower_cap_(g_), upper_cap_(g_), cost_(g_),
          flow_(g_) {}


private:
    CLASS_ARG(LemonGraph, g);                                ///< Lemon graph
    CLASS_ARG(Node, s);                                      ///< Flow source node
    CLASS_ARG(Node, t);                                      ///< Flow target node
    CLASS_ARG(LemonGraph::ArcMap<FlowIntType>, lower_cap);   ///< lower bound of capacity
    CLASS_ARG(LemonGraph::ArcMap<FlowIntType>, upper_cap);   ///< upper bound of capacity
    CLASS_ARG(LemonGraph::ArcMap<FlowIntType>, cost);        ///< Arc cost map
    CLASS_ARG(std::vector<Node>, left_nodes);                ///< Clock groups
    CLASS_ARG(std::vector<Node>, right_nodes);               ///< Clock regions
    CLASS_ARG(std::vector<Arc>, left_arcs);          ///< Arcs between source/right and left/target
    CLASS_ARG(std::vector<Arc>, right_arcs);         ///< Arcs between source/right and le)ft/target
    CLASS_ARG(std::vector<Arc>, mid_arcs);           ///< Arcs between left and right
    CLASS_ARG(LemonMinCostMaxFlowAlgorithm, flow);   ///< Lemon min-cost-max-flow object
    CLASS_ARG(FlowIntType, flow_supply) = 0;   ///< Flow supply, should be equal to ((total node
                                               ///< area or count) * constant)
    CLASS_ARG(FlowIntType, flow_capacity) = 0;   ///< Flow capacity, should be equal to (total clock
                                                 ///< region area or site count)
    CLASS_ARG(MidArcNodePairs,
              mid_arc_node_pairs);   ///< Node pair (left index, right index) for midArc
    CLASS_ARG(CrIdToMidArcIdxArray, cr_id2_mid_arc_idx_array);   ///< Index array of midArcs that
                                                                 ///< connects to each clock region
    CLASS_ARG(CkNetIdToMidArcIdxArray,
              ck_id2mid_arc_idx_array);   ///< Index array of midArcs that belong to each clock
                                          ///< nets

public:
    LemonMinCostMaxFlowAlgorithm::ProblemType run();
};

class MinCostFlowGraph {
    SHARED_CLASS_ARG(MinCostFlowGraph, impl);

public:
    FORWARDED_METHOD(g)
    FORWARDED_METHOD(s)
    FORWARDED_METHOD(t)
    FORWARDED_METHOD(lower_cap)
    FORWARDED_METHOD(upper_cap)
    FORWARDED_METHOD(left_nodes)
    FORWARDED_METHOD(right_nodes)
    FORWARDED_METHOD(left_arcs)
    FORWARDED_METHOD(right_arcs)
    FORWARDED_METHOD(mid_arcs)
    FORWARDED_METHOD(flow)
    FORWARDED_METHOD(flow_supply)
    FORWARDED_METHOD(flow_capacity)
    FORWARDED_METHOD(cr_id2_mid_arc_idx_array)
    FORWARDED_METHOD(ck_id2mid_arc_idx_array)
    FORWARDED_METHOD(mid_arc_node_pairs)
    FORWARDED_METHOD(cost)
    FORWARDED_METHOD(run)
};

MAKE_SHARED_CLASS(MinCostFlowGraph)

class MinCostFlowGraphWrapperImpl : public std::enable_shared_from_this<MinCostFlowGraphWrapperImpl> {
public:
    using IndexType   = int32_t;
    using Index2Graph = std::unordered_map<IndexType, MinCostFlowGraph>;

private:
    CLASS_ARG(MinCostFlowGraph, packed_group_graph);
    CLASS_ARG(Index2Graph, single_ele_group_graphs_map);

public:
    int32_t run();
};

class MinCostFlowGraphWrapper {
private:
    std::shared_ptr<MinCostFlowGraphWrapperImpl> impl_;

public:
    explicit MinCostFlowGraphWrapper(std::shared_ptr<MinCostFlowGraphWrapperImpl> impl)
        : impl_(std::move(impl)) {}

    FORWARDED_METHOD(run)
    FORWARDED_METHOD(packed_group_graph)
    FORWARDED_METHOD(single_ele_group_graphs_map)
};

MAKE_SHARED_CLASS(MinCostFlowGraphWrapper)

}   // namespace clock_network_planner
