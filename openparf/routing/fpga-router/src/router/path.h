#ifndef PATH_H
#define PATH_H

#include "database/pin.h"
#include "utils/utils.h"
#include <memory>
namespace router {

class PathNode {
public:
    PathNode(){}
    PathNode(int pinId);
    PathNode(int pinId, std::shared_ptr<PathNode> prev, COST_T cost);
    PathNode(int pinId, std::shared_ptr<PathNode> prev, COST_T cost, COST_T pred);
    PathNode(int pinId, std::shared_ptr<PathNode> prev, COST_T cost, COST_T pred, COST_T delay)
        : headPinId(pinId), prevNode(prev), nodeCost(cost), predCost(pred), nodeDelay(delay) {}

    std::shared_ptr<PathNode> getPrevNode() { return prevNode; }
    COST_T getCost() { return nodeCost; }
    int getHeadPin() { return headPinId; }

    bool operator<(const PathNode &p) const {
        return astarFac * p.predCost + p.nodeCost < astarFac * predCost + nodeCost;
    }

    friend class Pathfinder;
    friend class AStarPathfinder;

    static COST_T astarFac; 
private:
    int headPinId;
    std::shared_ptr<PathNode> prevNode;
    COST_T nodeCost;
    COST_T predCost;
    COST_T nodeDelay;
};

} // namespace router

#endif //PATH_H
