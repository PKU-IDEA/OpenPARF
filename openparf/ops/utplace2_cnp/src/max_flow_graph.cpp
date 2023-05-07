#include "max_flow_graph.h"
namespace clock_network_planner {

bool MaxFlowGraphImpl::run() {
    flow_.source(s_).target(t_).capacityMap(cap_map_);
    flow_.run();
    return flow_.flowValue() == flow_supply_;
}

}   // namespace clock_network_planner