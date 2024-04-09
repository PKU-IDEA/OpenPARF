#include "pathfinder.h"
#include <queue>
#include <memory>
#include <limits>

namespace router {
    RouteTree Pathfinder::routetree;
    std::vector<COST_T> Pathfinder::cost;
    std::vector<int> Pathfinder::prev;
    std::vector<int64_t> Pathfinder::visited;
    int Pathfinder::iter = 0;
    int Pathfinder::routedSinks = 0;
    std::vector<std::shared_ptr<TreeNode>> Pathfinder::treeNodes;

    bool Pathfinder::checkCanExpand(int nodeId) {
        auto pos = graph->getPosLow(nodeId), posHigh = graph->getPosHigh(nodeId);
        if (net->guide.start_x <= posHigh.X() && pos.X() <= net->guide.end_x && net->guide.start_y <= posHigh.Y() && pos.Y() <= net->guide.end_y)
            return true;
        return false;
    }


    RouteStatus Pathfinder::run() {
        //This Function is initially left unused
        
        return SUCCESS;

    }
} // namespace router