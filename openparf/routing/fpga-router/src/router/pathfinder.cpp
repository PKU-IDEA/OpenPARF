#include "pathfinder.h"
#include <queue>
#include <memory>
#include <limits>

namespace router {
    RouteTree Pathfinder::routetree;
    std::vector<COST_T> Pathfinder::cost;
    std::vector<COST_T> Pathfinder::delay;
    std::vector<int> Pathfinder::prev;
    std::vector<int64_t> Pathfinder::visited;
    int Pathfinder::iter = 0;
    int Pathfinder::routedSinks = 0;
    std::vector<std::shared_ptr<TreeNode>> Pathfinder::treeNodes;
    bool Pathfinder::isTimingDriven;
    bool Pathfinder::reverseSortOrder;

    bool Pathfinder::checkCanExpand(int nodeId) {
        auto pos = graph->getPos(nodeId), posHigh = graph->getPosHigh(nodeId);
        if (net->useGlobalResult()) {
            INDEX_T xlow = pos.X(), ylow = pos.Y(); 
            INDEX_T xhigh = posHigh.X(), yHigh = posHigh.Y();
            for (INDEX_T x = xlow; x <= xhigh; x++)
                for (INDEX_T y = ylow; y <= yHigh; y++) 
                    if (net->globalRouteResult.find(std::make_pair(x, y)) != net->globalRouteResult.end())
                        return true;         
        }
        else if (!net->useGlobalResult() && net->guide.start_x <= posHigh.X() && pos.X() <= net->guide.end_x && net->guide.start_y <= posHigh.Y() && pos.Y() <= net->guide.end_y)
            return true;
        return false;
    }

    RouteStatus Pathfinder::run() {
        
        // std::cout << module->getName() << std::endl;
        // BoundingBoxBuilder builder(net, module->getSubModule("CORE_A0"));
        // std::shared_ptr<RouteGraph> graph = builder.run();
        // std::cout << "Build Finished" << std::endl;

        // cost.assign(graph->getVertexNum(), std::numeric_limits<COST_T>::max());
        // prev.assign(graph->getVertexNum(), -1);

        // auto cmp = [](const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) {
        //     return *a < *b;
        // };

        // std::priority_queue<std::shared_ptr<PathNode>, std::vector<std::shared_ptr<PathNode>>, decltype(cmp)> q(cmp);

        // auto addNode = [&](const std::shared_ptr<PathNode> &node) {
        //     if (cost[node->headPinId] > node->nodeCost) {
        //         q.push(node);
        //         cost[node->headPinId] = node->nodeCost;
        //         prev[node->headPinId] = node->prevNode->headPinId;
        //     }
        // };
        // int sourceId;
        // int vertexNum = graph->getVertexNum();
        // for (int i = 0; i < vertexNum; i++) {
        //     if (graph->getOriginIdx(i) == net->getSource()) {
        //         sourceId = i;
        //         break;
        //     }
        // }
        // cost[sourceId] = 0;
        // q.push(std::shared_ptr<PathNode>(new PathNode(sourceId, nullptr, 0)));
        // int remainPins = net->getSinkSize();

        // std::vector<bool> visited(graph->getVertexNum(), 0);
        // std::vector<int>  sinkVertex;
        // while (!q.empty() && remainPins) {
        //     std::shared_ptr<PathNode> head = q.top();
        //     q.pop();

        //     if (visited[head->headPinId]) continue;
        //     visited[head->headPinId] = 1;
            
        //     // std::cout << head->getHeadPin()->getName() << "cost =" << head->nodeCost << std::endl;
            
        //     if (net->isSink(graph->getOriginIdx(head->headPinId))) {
        //         std::shared_ptr<PathNode> now = head;
        //         sinkVertex.push_back(head->headPinId);
        //         // bool isleaf = true;
        //         while (now != nullptr) {
        //             // std::cout << "NOW Pin is " << now->getHeadPin()->getName() << " PrevID = " << prev[now->headPinId] << std::endl; 
        //             if (!net->isSink(graph->getOriginIdx(head->headPinId))) visited[now->headPinId] = 0;
        //             // std::shared_ptr<database::Pin> prevPin = (now->getPrevNode() == nullptr ? nullptr : now->getPrevNode()->getHeadPin());
        //             // routetree.addNode(now->getHeadPin(), net, prevPin, isleaf);
        //             addNode(std::shared_ptr<PathNode>(new PathNode(now->headPinId, now->getPrevNode(), 0)));
        //             cost[now->headPinId] = 0;
        //             now = now->prevNode;
        //             // isleaf = false;
        //         }
                
        //         remainPins--;
        //         // continue;
        //     }

        //     int vertexDegree = graph->getVertexDegree(head->headPinId);
        //     int vertexIdx = head->headPinId;
        //     // std::cout << vertexIdx << ' ' << vertexDegree << std::endl;
        //     for (int i = 0; i < vertexDegree; i++) {
        //         int nextVertexIdx = graph->getEdge(vertexIdx, i);
        //         // std::cout << nextVertexIdx << std::endl;
        //         COST_T cost = graph->getEdgeCost(vertexIdx, i);
        //         addNode(std::shared_ptr<PathNode>(new PathNode(nextVertexIdx, head, head->getCost() + cost + graph->getVertexCost(nextVertexIdx))));
        //     }
        // }  
        
        // std::cout << "PathFinder Finish" << std::endl;

        // if (remainPins != 0) 
        //     return FAILED;

        // RouteStatus result = SUCCESS;
        // // std::cout << "sinks.size = " << net->sinks.size() << std::endl;
        // for (auto sink : sinkVertex) {
        //     int id = sink;
        //     bool isLeaf = true;
        //     while (id != -1) {
        //         int prevId = prev[id];
        //         int originId = graph->getOriginIdx(id);
        //         int originPrevId = graph->getOriginIdx(prevId);
        //         if (routetree.getNodeNet(originId) == net) break;
        //         if (routetree.getNodeNet(originId) != nullptr && routetree.getNodeNet(originId)->getRouteStatus() == SUCCESS) {
        //             routetree.getNodeNet(originId)->setRouteStatus(UNROUTED);
        //             globalGraph->addVertexCost(originId);
        //             result = UNROUTED;
        //         } else {
        //             routetree.addNode(originId, net, originPrevId, isLeaf);
        //         }
        //         // net->localRouteTree->addNode(pin, net, prevPin, isLeaf);
        //         id = prevId;
        //         isLeaf = false;
        //     }
        // }
        
        return SUCCESS;

    }
} // namespace router