#include "cr_id2ck_sink_set_factory.h"
#include "util/message.h"
#include <fstream>

namespace clock_network_planner {

    CrId2CkSinkSet
    CrId2CkSinkSetFactory::collectGroupAssignmentClockDistribution(GroupArrayWrapper group_array_wrapper,
                                                                   MinCostFlowGraphWrapper mcf_graph_wrapper) {
        using IndexType = MinCostFlowGraphImpl::IndexType;
        auto cr_id2ck_sink_set = detail::makeCrId2CkSinkSet();
        cr_id2ck_sink_set.clear();
        int32_t cr_num = mcf_graph_wrapper.packed_group_graph().right_nodes().size();
        cr_id2ck_sink_set.resize(cr_num);
        std::ofstream of;
        auto insert_functor = [&of, &cr_id2ck_sink_set](AssignmentGroupArray &group_array, MinCostFlowGraph graph) {
            for (int32_t mid_arc_id = 0; mid_arc_id < graph.mid_arcs().size(); mid_arc_id++) {
                auto &arc = graph.mid_arcs().at(mid_arc_id);
                auto &flow_node_pair = graph.mid_arc_node_pairs().at(mid_arc_id);
                auto &group_id = flow_node_pair.first;
                auto &cr_id = flow_node_pair.second;
                if (group_id == std::numeric_limits<IndexType>::max()
                    || graph.flow().flow(arc) == 0) {
                    continue;
                }
                auto &group = group_array.at(group_id);
                for (auto &ck_id : group.ck_sets()) {
                    cr_id2ck_sink_set.at(cr_id).insert(ck_id);
                }
                if(graph.cost()[arc]!=0){
                    of << group.inst_ids()[0] << " " << cr_id << " " << graph.cost()[arc] << std::endl;
                }
            }
            of.close();
        };
        of.open("mcf_result_0.txt");
        insert_functor(group_array_wrapper.packed_group_array(),
                       mcf_graph_wrapper.packed_group_graph());
        for (auto &iter : group_array_wrapper.single_ele_group_array_map()) {
            openparfAssert(mcf_graph_wrapper.single_ele_group_graphs_map().find(iter.first) !=
                           mcf_graph_wrapper.single_ele_group_graphs_map().end());
            std::stringstream ss;
            ss << iter.first;
            of.open("mcf_result_"+std::string(ss.str())+".txt");
            insert_functor(iter.second, mcf_graph_wrapper.single_ele_group_graphs_map()[iter.first]);
        }
        return cr_id2ck_sink_set;
    }
}