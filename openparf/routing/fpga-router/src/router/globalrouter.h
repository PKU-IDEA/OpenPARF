#ifndef GLOBALROUTER_H
#define GLOBALROUTER_H

#include "utils/utils.h"
#include "globalroutegraph.h"
#include "routegraph.h"
#include "globalroutetree.h"

#include <memory>
#include <vector>
namespace router {
    class GlobalRouter {
    public:
        GlobalRouter() {}
        GlobalRouter(std::shared_ptr<RouteGraph> _graph);

        void run(std::vector<std::shared_ptr<Net>>& netlist);
    private:
        RouteStatus route(std::shared_ptr<Net> net);
        RouteStatus routeSinglePath(std::shared_ptr<Net> net, int sink, std::vector<int>& sources, bool canOverflow);

        std::shared_ptr<RouteGraph> graph;
        std::shared_ptr<GlobalRouteGraph> globalGraph;

        GlobalRouteTree globalRouteTree;
        std::vector<std::shared_ptr<GlobalTreeNode> > globalTreeNodes; 
    };
}


#endif //GLOBALROUTER_H