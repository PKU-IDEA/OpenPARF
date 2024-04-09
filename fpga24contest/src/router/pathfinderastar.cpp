#include "pathfinderastar.h"
#include "../thirdparty/vprheap/binary_heap.h"
#include "thirdparty/taskflow/taskflow.hpp"

#include <queue>
#include <memory>
#include <unordered_set>
#include <stack>
#include <limits>
#include <algorithm>
#include <random>

#include <chrono>
using namespace std::chrono;

namespace router {

    RouteStatus AStarPathfinder::run() {
        net->addRerouteTime();
        // if (iter >= 50 || RouteGraph::presFac >= 1e11) highFanoutTrick = false;
        highFanoutTrick = false;

        if (net->getRouteStatus() != UNROUTED) {
            std::cerr << "[ERROR] routing net status not UNROUTED" << std::endl;
            exit(0);
        }

        // if (net->getName() == "net_10367") {
        //     std::cout << "name " << net->getName() << " steinerpoints size " << net->getGlobalSteinerPoints().size() << " result size " << net->getGlobalRouteResult().size() << " subtrees size " << net->subTreesInOrder.size() << "net root " << net->getSource() << std::endl;

        //     std::cout << "global route result" << std::endl;
        //     for (auto r : net->getGlobalRouteResult()) {
        //         std::cout << r.first << " " << r.second << std::endl;
        //     }
        // }

        // if (net->getSinkSize() >= highFanoutThres) {
        auto& sinks = net->getSinks();
        auto posSource = graph->getPos(net->getSource());
        auto cmp = [&](int a, int b) {
            auto posa = graph->getPos(a);
            auto posb = graph->getPos(b);
            return abs(posa.X() - posSource.X()) + abs(posa.Y() - posSource.Y()) < abs(posb.X() - posSource.X()) + abs(posb.Y() - posSource.Y());
        };
        std::sort(sinks.begin(), sinks.end(), cmp);
        // }

        // BoundingBoxBuilder builder(net, module->getSubModule("CORE_A0"));
        // std::shared_ptr<RouteGraph> graph = builder.run();
        // for incremental routing ???
        std::queue<std::shared_ptr<TreeNode>> routeTreeQueue;
        routeTreeQueue.push(routetree.getNetRoot(net));
        while (!routeTreeQueue.empty()) {
            auto node = routeTreeQueue.front();
            routeTreeQueue.pop();
            if (node->firstChild == nullptr && node->nodeId != net->getSource() && !net->isSink(node->nodeId)) {
                std::cout << "[ERROR OCCURED] " << net->getName() << " have open vertex " << node->nodeId;
                getchar();
            }
            // if (net->getName() == "net_71560" && iter)
            //     std::cout << "NodeId: " << node->nodeId << " Node: " << graph->getVertexByIdx(node->nodeId) << std::endl;
            startVerticesOrigin.push_back(node->nodeId);
            treeNodes[node->nodeId] = node;
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                // if (net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]") 
                //     std::cout << node->nodeId << "->" << child->nodeId << std::endl;
                routeTreeQueue.push(child);
            }

            // std::cout << "routeTreeQueue nodeId " << node->nodeId << std::endl;

        }

        int remainPins = 0;
        for (int i = 0; i < net->getSinkSize(); i++) {
            int pin = net->getSinkByIdx(i);

            // std::cout << "Net " << net->getName()
            //     << " sink " << graph->getVertexByIdx(pin)->getName()
            //     << " X " << graph->getVertexByIdx(pin)->getPos().X()
            //     << " Y " << graph->getVertexByIdx(pin)->getPos().Y()
            //     << std::endl;

            if (treeNodes[pin] == nullptr || treeNodes[pin]->net != net) {
                // if (routeSingleSink(i) == FAILED) return FAILED;
                // this pin is unrouted (need to route) or congested (also need to route)
                // predictor.add(pin);
                // if (iter < 50 && RouteGraph::presFac < 1e11) highFanoutTrick = true;
                if (routeSingleSink(i) == FAILED) {
                    // std::cout << net->getName() << ' ' << i << ' ' << pin << std::endl;
                    // auto guide = net->getGuide();
                    // std::cout << "BBox: " << guide.start_x << ' ' << guide.start_y << ' ' << guide.end_x << ' ' << guide.end_y << std::endl;
                    // std::cout << "route idx " << pin << " wire " << graph->getVertexByIdx(pin)->getName() << " FAILED X: " << graph->getPos(pin).X() << " Y: " << graph->getPos(pin).Y() << std::endl; 
                    if (net->getSinkSize() >= highFanoutThres) {
                        highFanoutTrick = false;
                        if (routeSingleSink(i) == FAILED) {
                            // std::cout << "net " << net->getName() << " failed, routeRegion " << net->globalRouteResult.size() << " sink num " << net->getSinkSize() << " pin name " << graph->getVertexByIdx(pin)->getName() << std::endl;
                            if (net->getSinkSize() < 2) {
                                std::cout << "source pos " << graph->getPos(net->getSource()).X() << " " << graph->getPos(net->getSource()).Y() 
                                    << " posLow " << graph->getPosLow(net->getSource()).X() << " " << graph->getPosLow(net->getSource()).Y()
                                    << " posHigh " << graph->getPosHigh(net->getSource()).X() << " " << graph->getPosHigh(net->getSource()).Y()
                                    << std::endl;
                                std::cout << "sink pos " << graph->getPos(pin).X() << " " << graph->getPos(pin).Y() 
                                    << " posLow " << graph->getPosLow(pin).X() << " " << graph->getPosLow(pin).Y()
                                    << " posHigh " << graph->getPosHigh(pin).X() << " " << graph->getPosHigh(pin).Y()
                                    << std::endl;
                                // for (auto g : net->globalRouteResult) {
                                //     std::cout << g.first << " " << g.second << std::endl;
                                // }
                            }
                            std::cout << "route idx " << pin << " wire " << graph->getVertexByIdx(pin)->getName() << " FAILED X: " << graph->getPos(pin).X() << " Y: " << graph->getPos(pin).Y() << std::endl; 
                            std::cout << net->getName() << ' ' << net->getSinkSize() << std::endl;
                                auto bbox =  net->guide;
                                std::cout << bbox.start_x << ' ' << bbox.start_y << ' ' << bbox.end_x << ' ' << bbox.end_y << std::endl;
                            routeTreeQueue.push(routetree.getNetRoot(net));
                            while (!routeTreeQueue.empty()) {
                                auto node = routeTreeQueue.front();
                                routeTreeQueue.pop();
                                if (node->firstChild == nullptr && node->nodeId != net->getSource() && !net->isSink(node->nodeId)) {
                                    std::cout << "[ERROR OCCURED] " << net->getName() << " have open vertex " << node->nodeId;
                                    getchar();
                                }
                                // if (net->getName() == "net_71560" && iter)
                                //     std::cout << "NodeId: " << node->nodeId << " Node: " << graph->getVertexByIdx(node->nodeId) << std::endl;
                                // startVerticesOrigin.push_back(node->nodeId);
                                // treeNodes[node->nodeId] = node;
                                std::cout << graph->getVertexByIdx(node->nodeId)->getName() << ' ' << node->nodeId << ' ' << graph->getPos(node->nodeId).X() << ' '  << graph->getPos(node->nodeId).Y() << ' ';
                                if (node->father != nullptr)
                                std::cout << graph->getVertexByIdx(node->father->nodeId)->getName() << std::endl;
                                else std::cout << std::endl;
                                for (auto child = node->firstChild; child != nullptr; child = child->right) {
                                    // if (net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]") 
                                    //     std::cout << node->nodeId << "->" << child->nodeId << std::endl;
                                    routeTreeQueue.push(child);
                                }
                            }
                            // exit(0);
                            return FAILED;
                        }
                    }
                    // getchar();
                    else {
                        // std::cout << "net " << net->getName() << " failed, routeRegion " << net->globalRouteResult.size() << " sink num " << net->getSinkSize() << " pin name " << graph->getVertexByIdx(pin)->getName() << std::endl;
                        if (net->getSinkSize() < 2) {
                            std::cout << "source pos " << graph->getPos(net->getSource()).X() << " " << graph->getPos(net->getSource()).Y() 
                                << " posLow " << graph->getPosLow(net->getSource()).X() << " " << graph->getPosLow(net->getSource()).Y()
                                << " posHigh " << graph->getPosHigh(net->getSource()).X() << " " << graph->getPosHigh(net->getSource()).Y()
                                << std::endl;
                            std::cout << "sink pos " << graph->getPos(pin).X() << " " << graph->getPos(pin).Y() 
                                << " posLow " << graph->getPosLow(pin).X() << " " << graph->getPosLow(pin).Y()
                                << " posHigh " << graph->getPosHigh(pin).X() << " " << graph->getPosHigh(pin).Y()
                                << std::endl;
                            // for (auto g : net->globalRouteResult) {
                            //     std::cout << g.first << " " << g.second << std::endl;
                            // }
                        }
                        std::cout << "route idx " << pin << " wire " << graph->getVertexByIdx(pin)->getName() << " FAILED X: " << graph->getPos(pin).X() << " Y: " << graph->getPos(pin).Y() << std::endl; 
                        std::cout << net->getName() << ' ' << net->getSinkSize() << std::endl;
                        auto bbox =  net->guide;
                        std::cout << bbox.start_x << ' ' << bbox.start_y << ' ' << bbox.end_x << ' ' << bbox.end_y << std::endl;
                        routeTreeQueue.push(routetree.getNetRoot(net));
                        while (!routeTreeQueue.empty()) {
                            auto node = routeTreeQueue.front();
                            routeTreeQueue.pop();
                            if (node->firstChild == nullptr && node->nodeId != net->getSource() && !net->isSink(node->nodeId)) {
                                std::cout << "[ERROR OCCURED] " << net->getName() << " have open vertex " << node->nodeId;
                                getchar();
                            }
                            // if (net->getName() == "net_71560" && iter)
                            //     std::cout << "NodeId: " << node->nodeId << " Node: " << graph->getVertexByIdx(node->nodeId) << std::endl;
                            // startVerticesOrigin.push_back(node->nodeId);
                            // treeNodes[node->nodeId] = node;
                            std::cout << graph->getVertexByIdx(node->nodeId)->getName() << ' ' << node->nodeId << ' ' << graph->getPos(node->nodeId).X() << ' '  << graph->getPos(node->nodeId).Y() << ' ';
                            if (node->father != nullptr)
                            std::cout << graph->getVertexByIdx(node->father->nodeId)->getName() << std::endl;
                            else std::cout << std::endl;
                            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                                // if (net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]") 
                                //     std::cout << node->nodeId << "->" << child->nodeId << std::endl;
                                routeTreeQueue.push(child);
                            }
                        }
                        // getchar();
                        return FAILED;
                    }
                }
            } else {
                routedSinks++;
            }
            // sinkVertex.insert(pin);
        }


        auto isSteiner = [](const std::shared_ptr<TreeNode> &node) {

            int numAdjNodes = 0;
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                numAdjNodes++;
            }

            return numAdjNodes > 1;
        };


        // print steiner point in detialed routetree
        std::queue<std::shared_ptr<TreeNode>> printSteinerQ;
        printSteinerQ.push(routetree.getNetRoot(net));

        while (!printSteinerQ.empty())
        {
            auto node = printSteinerQ.front();
            printSteinerQ.pop();

            // if (isSteiner(node)) {
            //     std::cout << "Steiner type " << graph->getVertexType(node->nodeId) << " id " << node->nodeId << " posX " << graph->getPos(node->nodeId).X() << " posY " << graph->getPos(node->nodeId).Y() << " name " << graph->getVertexByIdx(node->nodeId)->getName() 
            //     << " degree " << graph->getVertexDegree(node->nodeId)
            //     << " reverseDegree " << graph->getVertexDegreeReverse(node->nodeId)
            //     << " GSWConnect " << graph->getVertexByIdx(node->nodeId)->getGSWConnectPin()
            //     << " GSWDirection " << graph->getVertexByIdx(node->nodeId)->getGSWConnectDirection()
            //     << " GSWConnectLength " << graph->getVertexByIdx(node->nodeId)->getGSWConnectLength()
            //     << std::endl;
            // }
        
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                printSteinerQ.push(child);
            }
        
        }



        return SUCCESS;
    }

    COST_T AStarPathfinder::predictNaive (int vertex, int target) {
        if (!RouteGraph::useAStar) {
            return 0;
        } else {
            auto posS = graph->getPosLow(vertex);
            auto posT = graph->getPosLow(target);
            return (abs(posS.X() - posT.X()) + abs(posS.Y() - posT.Y())) * baseCost;
        }
    }

    RouteStatus AStarPathfinder::routeSingleSink(int targetIdx) {
        int target = net->getSinkByIdx(targetIdx);
        // std::cout << "routing target " << graph->getVertexByIdx(target)->getName() << " net " << net->getName() << std::endl;
        // std::cout << "target pos: " << graph->getVertexByIdx(target)->getPos().X() << ' ' << graph->getVertexByIdx(target)->getPos().Y() << std::endl;
        // if (target != 2932075) return SUCCESS;
        bool isIndirect = net->isIndirect(); // INDIRECT NETS ONLY USE INT TILES
        auto cmp = [](const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) {
            return *a < *b;
        };

        // COST_T rerouteCriticality = RouteGraph::minRerouteCriticality;
        COST_T rerouteCriticality = 0;
        COST_T rnodeCostWeight = 1 - rerouteCriticality;
        COST_T shareWeight = std::pow(rnodeCostWeight, RouteGraph::shareExponent);
        COST_T rnodeWLWeight = rnodeCostWeight * (1 - RouteGraph::wirelengthWeight);
        COST_T estWlWeight = rnodeCostWeight * RouteGraph::wirelengthWeight;
        // COST_T estWlWeight = rnodeCostWeight;
        COST_T dlyWeight = rerouteCriticality * (1 - RouteGraph::timingWeight) / 100;
        COST_T estDlyWeight = rerouteCriticality * RouteGraph::timingWeight;

        // std::priority_queue<std::shared_ptr<PathNode>, std::vector<std::shared_ptr<PathNode>>, decltype(cmp)> q(cmp);
        BinaryHeap heap;
        heap.init_heap(graph);
        
        int64_t visitID = 1LL * net->netId * 1e9 + targetIdx * 10 + highFanoutTrick;

        auto addNode = [&](int vertexIdx, int prevIdx, COST_T nodeCost, COST_T predCost) {
            if (predCost != 1e9) {
            // if (net->getName() == "net_118568" || net->getName() == "net_116700")
            // if (RouteGraph::debugging)
            // if (target == 10697333)
                // std::cout << "trying to add Node" << vertexIdx << ' ' << graph->getVertexByIdx(vertexIdx)->getName() << " cost: " << nodeCost << " predCost: " << predCost << " pos: " << graph->getPos(vertexIdx).X() << ' ' <<
                // graph->getPos(vertexIdx).Y() << ' ' << graph->getPosHigh(vertexIdx).X() << ' ' << graph->getPosHigh(vertexIdx).Y() <<
                // " check result: " << checkCanExpand(vertexIdx) << " prev cost: " << cost[vertexIdx] << std::endl;
            // if (net->getName() == "u_calc/dropSpin/absorb/squareRoot/op__250[51]" && graph->getVertexByIdx(vertexIdx)->getName() == "CLK_LEAF_SITES_9_CLK_LEAF")
            //     std::cout << isIndirect << ' ' << graph->getVertexType(vertexIdx) << std::endl;
            if (graph->getVertexType(vertexIdx) == VertexType::NETSINK && (vertexIdx != net->getSource() && vertexIdx != target)) return;
            if (isIndirect && graph->getVertexType(vertexIdx) != VertexType::INTTILE && graph->getVertexType(vertexIdx) != VertexType::NETSINK && graph->getVertexType(vertexIdx) != VertexType::SOURCE) return;
            if (checkCanExpand(vertexIdx) && ((visited[vertexIdx] != visitID) || (visited[vertexIdx] == visitID && cost[vertexIdx] > nodeCost))) {                
                    // q.push(node);
                    net->inQueueCnt++;
                    visited[vertexIdx] = visitID;
                    cost[vertexIdx] = nodeCost;
                    // if (prevNode != nullptr)
                    prev[vertexIdx] = prevIdx;
                    t_heap* heapNode = heap.alloc();
                    heapNode->cost = nodeCost + predCost * PathNode::astarFac;
                    heapNode->backward_path_cost = nodeCost;
                    heapNode->set_prev_node(prev[vertexIdx]);
                    heapNode->index = vertexIdx;
                    heap.add_to_heap(heapNode);
                }
            }
        };

        if (net->getSinkSize() < highFanoutThres || !highFanoutTrick) {
            for (auto startVertex : startVerticesOrigin) {
                if (!(checkCanExpand(startVertex) && ((visited[startVertex] != visitID) || (visited[startVertex] == visitID && cost[startVertex] > 0)))) {
                    std::cout << "bug appear!" << ' ' << net->getName() << ' ' << target << ' ' << startVertex << std::endl;
                    std::cout << checkCanExpand(startVertex) << ' ' << visited[startVertex] << ' ' << visitID << ' ' << cost[startVertex] << std::endl;
                    std::cout << graph->getVertexByIdx(startVertex)-> getName() << ' ' << graph->getPos(startVertex).X() << ' ' << graph->getPos(startVertex).Y() << ' ' << graph->getPosHigh(startVertex).X() << graph->getPosHigh(startVertex).Y() << std::endl;
                    std::cout << graph->getVertexByIdx(target)->getName() << ' ' << graph->getPos(target).X() << ' ' << graph->getPos(target).Y() << std::endl;
                    std::cout << visited[startVertex] << ' ' << visitID << ' ' << cost[startVertex] << std::endl; 
                    getchar();
                }
                addNode(startVertex, -1, 0, estWlWeight *  predictNaive(startVertex, target));
            }
        }
        else {
            bool flag = false;
            auto posT = graph->getPosLow(target);
            for (auto startVertex : startVerticesOrigin) {
                if (!(checkCanExpand(startVertex) && ((visited[startVertex] != visitID) || (visited[startVertex] == visitID && cost[startVertex] > 0)))) {
                    std::cout << "bug appear!" << ' ' << net->getName() << ' ' << target << ' ' << startVertex << std::endl;
                    std::cout << checkCanExpand(startVertex) << ' ' << visited[startVertex] << ' ' << visitID << ' ' << cost[startVertex] << std::endl;
                    std::cout << graph->getVertexByIdx(startVertex)-> getName() << ' ' << graph->getPos(startVertex).X() << ' ' << graph->getPos(startVertex).Y() << ' ' << graph->getPosHigh(startVertex).X() << graph->getPosHigh(startVertex).Y() << std::endl;
                    std::cout << graph->getVertexByIdx(target)->getName() << ' ' << graph->getPos(target).X() << ' ' << graph->getPos(target).Y() << std::endl;
                    std::cout << visited[startVertex] << ' ' << visitID << ' ' << cost[startVertex] << std::endl; 
                    getchar();
                }
                auto posS = graph->getPosLow(startVertex);
                if (abs(posS.X() - posT.X()) + abs(posS.Y() - posT.Y()) <= maxHighFanoutAddDist) {
                    addNode(startVertex, -1, 0, estWlWeight *  predictNaive(startVertex, target));
                    flag = true;
                }
            }
            if (!flag) {
                for (auto startVertex : startVerticesOrigin) {
                    addNode(startVertex, -1, 0, estWlWeight * predictNaive(startVertex, target));
                }
            }
        }

        while (!heap.is_empty_heap()) {
            auto head = heap.get_heap_head();
            net->outQueueCnt++;

            // if (head->nodeCost != cost[head->headPinId]) continue;
            // std::cout << head->headPinId << ' ' << head->nodeCost << ' ' << head->predCost << std::endl;  
            // if (net->getName() == "net_118568" || net->getName() == "net_116700")
            // if (RouteGraph::debugging)
            // if (target == 10697333) {
            // printf("Poping Vertex %d %s, Type: %d, cap = %d, cost = %g, predCost = %g, xlow = %d, ylow = %d, xhigh = %d, yhigh = %d\n"
            // , head->index, graph->getVertexByIdx(head->index)->getName().c_str(), (int)graph->getVertexType(head->index), graph->getVertexCap(head->index),
            // head->backward_path_cost, head->cost, graph->getPos(head->index).X(),
            // graph->getPos(head->index).Y(), graph->getPosHigh(head->index).X(), graph->getPosHigh(head->index).Y());
            // getchar();
            // }

            // if (net->getName() == "net_161786") {
            //     std::cout << "heap head pin " << head->index << " pos " << graph->getPos(head->index).X() << " " << graph->getPos(head->index).Y() 
            //         << std::endl;
            // }

            // std::cout << "Pop -> node: " << head->index 
            //     << " cost " << cost[head->index]
            //     << " wireName " << graph->getVertexByIdx(head->index)->getName()
            //     << " posX " << graph->getVertexByIdx(head->index)->getPos().X()
            //     << " posY " << graph->getVertexByIdx(head->index)->getPos().Y()
            //     << std::endl;

            if (head->index == target) {
                // std::cout << "Target " << target << "Found!" << std::endl;
                int now = head->index;
                std::stack<int> pathNodeIds;
                while (now != -1 && (treeNodes[now] == nullptr || treeNodes[now]->net != net)) {
                    // if (net->getName() == "net_82858" || net->getName() == "net_82857")
                    // if (RouteGraph::debugging)
                    //     std::cout << "Node Id: " << now << " pin Name: " << graph->getVertexByIdx(now)->getName() << std::endl;
                    pathNodeIds.push(now);
                    now = prev[now];
                }
                while (!pathNodeIds.empty()) {
                    int node = pathNodeIds.top();
                    pathNodeIds.pop();
                    // std::cout << "Path -> Net: " << net->getName()
                    //         << ", node: " << node 
                    //         << ", nodeType: " << graph->getVertexIC(node)
                    //         << ", cost: " << cost[node]
                    //         << ", prev: " << prev[node]
                    //         << ", posX: " << graph->getVertexByIdx(node)->getPos().X()
                    //         << ", posY: " << graph->getVertexByIdx(node)->getPos().Y()
                    //         << std::endl;
                    startVerticesOrigin.push_back(node);
                    treeNodes[node] = routetree.addNode(treeNodes[prev[node]], node, net);
                    // if (net->getName() == "u_calc/dropSpin/scattererReflector/squareRoot1_6/op__11_q[63]_i_190__0_n_0") {
                    //     std::cout << node << ' ' << graph->getVertexByIdx(node)->getName() << ' ' << routetree.getTreeNodeByIdx(node) << std::endl;
                    //     getchar();
                    // }
                }
                heap.free(head);
                return SUCCESS; 
            } 
            
            int vertexDegree = graph->getVertexDegree(head->index);
            int vertexIdx = head->index;
            // for (int i = 0; i < vertexDegree; i++) {
            //     int nextVertexIdx = graph->getEdge(vertexIdx, i);
            //     COST_T edgeCost = graph->getEdgeCost(vertexIdx, i);
            //     addNode(nextVertexIdx, vertexIdx, cost[vertexIdx] + edgeCost * graph->getVertexCost(nextVertexIdx), predictNaive(nextVertexIdx, target));

            //     // if (net->getName() == "net_10367") {
            //     //     std::cout << "Add edge node " << nextVertexIdx << " (" << graph->getPos(nextVertexIdx).X() << ", " << graph->getPos(nextVertexIdx).Y() << ") name " << graph->getVertexByIdx(nextVertexIdx)->getName() << " into net " << net->getName() << "'s root tree " << std::endl;
            //     // }
            // }
            for (int i = graph->getHeadOutEdgeIdx(vertexIdx); i != -1; i = graph->getEdge(i).preFromEdge) {
                // std::cout << i << std::endl;
                // getchar();
                auto& edge = graph->getEdge(i);
                int nextVertexIdx = edge.to;
                COST_T edgeCost = edge.cost;

                bool earlyTerminated = false;
                if (nextVertexIdx == target) {
                    if (graph->isVertexNotOccupied(nextVertexIdx) || (graph->getVertexIC(nextVertexIdx) == NODE_PINBOUNCE)) {
                        earlyTerminated = true;
                        heap.empty_heap();
                    }
                }

                COST_T pathCost = cost[vertexIdx] + rnodeCostWeight * graph->getVertexCost(nextVertexIdx);
                // COST_T pathCost = cost[vertexIdx] + rnodeCostWeight * graph->getVertexCost(nextVertexIdx) + rnodeWLWeight * edgeCost;
                // COST_T pathCost = cost[vertexIdx] + rnodeCostWeight * graph->getVertexCost(nextVertexIdx) + rnodeWLWeight * graph->getVertexLength(nextVertexIdx);

                // rnode.getUpstreamPathCost() + rnodeCostWeight * getNodeCost(childRnode, connection, countSourceUses, sharingFactor)
                //                 + rnodeLengthWeight * childRnode.getLength() / sharingFactor;

                // if (graph->getVertexByIdx(nextVertexIdx)->getName() == "CLE_CLE_M_SITE_0_HMUX") continue;

                // std::cout << "Push -> node: " << vertexIdx 
                //     << " wireName " << graph->getVertexByIdx(nextVertexIdx)->getName()
                //     << " intentCode " << graph->getVertexIC(nextVertexIdx)
                //     << " predCost " << cost[vertexIdx]
                //     << " nodeCost " << graph->getVertexCost(nextVertexIdx)
                //     << " totalPathCost " << pathCost
                //     << " rnodeCostWeight " << rnodeCostWeight
                //     << " nextVertexCost " << graph->getVertexCost(nextVertexIdx)
                //     << " nextVertexLength " << graph->getVertexLength(nextVertexIdx)
                //     << " posX " << graph->getVertexByIdx(nextVertexIdx)->getPos().X()
                //     << " posY " << graph->getVertexByIdx(nextVertexIdx)->getPos().Y()
                //     << std::endl;

                addNode(nextVertexIdx, vertexIdx, pathCost, estWlWeight * predictNaive(nextVertexIdx, target));
                if (earlyTerminated) break;
            }
            // std::cout << " ------------------- " << std::endl;
            heap.free(head);
        }
        return FAILED;
    }

    // int countSourceUses = childRnode.countConnectionsOfUser(connection.getNetWrapper());
    // float sharingFactor = 1 + sharingWeight* countSourceUses;
    // float newPartialPathCost = rnode.getUpstreamPathCost() + rnodeCostWeight * getNodeCost(childRnode, connection, countSourceUses, sharingFactor)
    //                         + rnodeLengthWeight * childRnode.getLength() / sharingFactor;
    // float newTotalPathCost = newPartialPathCost + rnodeEstWlWeight * distanceToSink / sharingFactor;


} // namespace router