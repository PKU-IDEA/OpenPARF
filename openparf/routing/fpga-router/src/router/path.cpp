#include "path.h"
#include <memory>
namespace router {

COST_T PathNode::astarFac = 1.2;

PathNode::PathNode(int pinId) {
    headPinId = pinId;
    prevNode = nullptr;
    nodeCost = 0;
    predCost = 0;
} 

PathNode::PathNode(int pinId, 
                   std::shared_ptr<PathNode> prev, 
                   COST_T cost) {
    headPinId = pinId;
    prevNode = prev;
    nodeCost = cost;
}


PathNode::PathNode(int pinId, 
                   std::shared_ptr<PathNode> prev, 
                   COST_T cost,
                   COST_T pred) {
    headPinId = pinId;
    prevNode = prev;
    nodeCost = cost;
    predCost = pred;
}

}
