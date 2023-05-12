#ifndef ASTAR_H
#define ASTAR_H

#include "pathfinder.h"
// #include "netconnect.h"

#include <set>

namespace router {
    
    class CostPredictor {
    public:
        virtual COST_T predict(int pin);
    };

    class NaivePredictor {
    public:
        NaivePredictor() {}
        NaivePredictor(std::shared_ptr<RouteGraph> _graph) : graph(_graph) {}
        COST_T predict(int pin);

        void add(int pin);
        void erase(int pin);

    protected:
        std::multiset<INDEX_T> xIndexs;
        std::multiset<INDEX_T> yIndexs;

        std::shared_ptr<RouteGraph> graph;
    };
    
    class AStarPathfinder : public Pathfinder {
    public:
        AStarPathfinder() {}
        AStarPathfinder(std::shared_ptr<Net> routenet, std::shared_ptr<RouteGraph> graph) : Pathfinder(routenet, graph) {}

        RouteStatus run();
        RouteStatus routeSingleSink(int targetIdx); 

        // RouteStatus routeNetConnect(NetConnect& netconnect);
        COST_T predict(int vertex, int target);
        COST_T timingDrivenPridict(int vertex, int target, COST_T critical);
        COST_T timingDrivenEdgeCost(int fromVertex, int edgeIdx, COST_T critical);

        bool checkPossiblePathVertex(int vertexId, int source, int sink);

        std::vector<int> startVertices;
        // std::unordered_map<int, std::shared_ptr<TreeNode>> visitedTreeNodes;
        bool highFanoutTrick = true;

    };
} // namespace router

#endif //ASTAR_H