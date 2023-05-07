#ifndef PATHFINDER_H
#define PATHFINDER_H

#include "path.h"
#include "routegraphbuilder.h"
#include "routegraph.h"
#include "net.h"
#include "routetree.h"
#include "database/module.h"
#include "utils/utils.h"

#include <unordered_map>

namespace router {
class Pathfinder {
public:

    Pathfinder(){}
    Pathfinder(std::shared_ptr<Net> routenet, std::shared_ptr<RouteGraph> _graph) { 
        net = routenet; 
        // graph = _graph->dumpLocalRouteGraph(net);
        graph = _graph;
    }
    bool checkCanExpand(int nodeId);

    virtual RouteStatus run();
    static RouteTree routetree;
    static std::vector<COST_T> cost;
    static std::vector<int> prev;
    static std::vector<int64_t> visited;
    static std::vector<COST_T> delay;
    static std::vector<std::shared_ptr<TreeNode>> treeNodes;

    static int iter;
    static int routedSinks;
    static bool isTimingDriven;
    static bool reverseSortOrder;

protected:
    std::shared_ptr<Net> net;
    std::shared_ptr<RouteGraph> graph;

    COST_T totalCost;
};

} // namespace router

#endif