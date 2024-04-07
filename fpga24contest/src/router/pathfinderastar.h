#ifndef ASTAR_H
#define ASTAR_H

#include "pathfinder.h"

#include <set>

namespace router {
    
    
    class AStarPathfinder : public Pathfinder {
    public:
        AStarPathfinder() {}
        AStarPathfinder(std::shared_ptr<Net> routenet, std::shared_ptr<RouteGraph> graph) : Pathfinder(routenet, graph) {}
        
        RouteStatus run();
        RouteStatus routeSingleSink(int targetIdx);
        COST_T predictNaive(int vertex, int target);

        std::vector<int> startVerticesOrigin; // use for original sequential routing
        // std::unordered_map<int, std::shared_ptr<TreeNode>> visitedTreeNodes;
        bool highFanoutTrick = true;

        // std::vector<std::set<int>> tempTracks;


    };
} // namespace router

#endif //ASTAR_H