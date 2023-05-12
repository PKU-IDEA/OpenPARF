#include "pathfinderastar.h"
#include "vprheap/binary_heap.h"

#include <queue>
#include <memory>
#include <unordered_set>
#include <stack>
#include <limits>
#include <algorithm>
namespace router {

    COST_T NaivePredictor::predict(int pin) {
        if (xIndexs.size() == 0) return 0;
        XY<INDEX_T> pos = graph->getPos(pin);
        INDEX_T x_pin = pos.X();
        INDEX_T y_pin = pos.Y();
        // std::cout << "x_pin = " << x_pin << " y_pin = " << y_pin << std::endl;

        COST_T predCost = 0;

        auto x_iter = xIndexs.lower_bound(x_pin);
        if (x_iter == xIndexs.end()) {
            x_iter--;
            predCost += x_pin - *x_iter;
        }
        else if (x_iter == xIndexs.begin()) {
            predCost += *x_iter - x_pin;
        }
        else {
            INDEX_T x1 = *x_iter;
            x_iter--;
            predCost += std::min(x1 - x_pin, x_pin - *x_iter);
        }

        auto y_iter = yIndexs.lower_bound(y_pin);
        if (y_iter == yIndexs.end()) {
            y_iter--;
            predCost += y_pin - *y_iter;
        }
        else if (y_iter == yIndexs.begin()) {
            predCost += *y_iter - y_pin;
        }
        else {
            INDEX_T y1 = *y_iter;
            y_iter--;
            predCost += std::min(y1 - y_pin, y_pin - *y_iter);
        }

        return predCost;
    }

    void NaivePredictor::add(int pin) {
        xIndexs.insert(graph->getPos(pin).X());
        yIndexs.insert(graph->getPos(pin).Y());
    }

    void NaivePredictor::erase(int pin) {
        // std::cout << "Erasing " << pin << ' ' << graph->getPos(pin).X() << ' ' << graph->getPos(pin).Y() << ' ' << xIndexs.size() << std::endl;
        xIndexs.erase(xIndexs.find(graph->getPos(pin).X()));
        yIndexs.erase(yIndexs.find(graph->getPos(pin).Y()));
    }

    RouteStatus AStarPathfinder::run() {
        net->addRerouteTime();
        if (iter >= 50 || RouteGraph::presFac >= 1e11) highFanoutTrick = false;

        if (net->getRouteStatus() != UNROUTED) {
            std::cerr << "[ERROR] routing net status not UNROUTED" << std::endl;
            exit(0);
        }

        // if (net->getSinkSize() >= highFanoutThres) {
        auto& sinks = net->getSinks();
        auto posSource = graph->getPos(net->getSource());
        auto cmp = [&](int a, int b) {
            auto posa = graph->getPos(a);
            auto posb = graph->getPos(b);
            if (reverseSortOrder)
                return abs(posa.X() - posSource.X()) + abs(posa.Y() - posSource.Y()) > abs(posb.X() - posSource.X()) + abs(posb.Y() - posSource.Y());
            else
                return abs(posa.X() - posSource.X()) + abs(posa.Y() - posSource.Y()) < abs(posb.X() - posSource.X()) + abs(posb.Y() - posSource.Y());
        };
        std::sort(sinks.begin(), sinks.end(), cmp);
        // }

        // BoundingBoxBuilder builder(net, module->getSubModule("CORE_A0"));
        // std::shared_ptr<RouteGraph> graph = builder.run();
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
            startVertices.push_back(node->nodeId);
            treeNodes[node->nodeId] = node;
            delay[node->nodeId] = node->nodeDelay;
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                // if (net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]")
                //     std::cout << node->nodeId << "->" << child->nodeId << std::endl;
                routeTreeQueue.push(child);
            }
        }

        int remainPins = 0;
        for (int i = 0; i < net->getSinkSize(); i++) {
            int pin = net->getSinkByIdx(i);
            if (treeNodes[pin] == nullptr || treeNodes[pin]->net != net) {
                // predictor.add(pin);
                if (iter < 50 && RouteGraph::presFac < 1e11)
                highFanoutTrick = true;
                if (routeSingleSink(i) == FAILED) {
                    // std::cout << net->getName() << ' ' << i << ' ' << graph->getVertexByIdx(pin)->getName() << std::endl;
                    // auto guide = net->getGuide();
                    // std::cout << "BBox: " << guide.start_x << ' ' << guide.start_y << ' ' << guide.end_x << ' ' << guide.end_y << std::endl;

                    // getchar();
                    if (net->getSinkSize() >= highFanoutThres) {
                        highFanoutTrick = false;
                        if (routeSingleSink(i) == FAILED) return FAILED;
                    }
                    else if (net->useGlobalResult()) {
                        net->useGlobalResult(false);
                        highFanoutTrick = false;
                        if (routeSingleSink(i) == FAILED) return FAILED;
                    }
                    else return FAILED;
                }
            }
            else routedSinks++;
            // sinkVertex.insert(pin);
        }

        return SUCCESS;
    }

    COST_T AStarPathfinder::predict (int vertex, int target) {
        if (!RouteGraph::useAStar) { return 0;}
        if (graph->getVertexType(vertex) == CHANX) {
            COST_T ret = 0;
            auto posS = graph->getPos(vertex), posSH = graph->getPosHigh(vertex);
            auto posT = graph->getPos(target);
            if (posT.Y() == posS.Y() || posT.Y() == posS.Y() - 1) {
                if (posT.X() < posS.X())
                    ret = ((posS.X() - posT.X() - 1) / 4 + 1) * 4;
                else if (posT.X() <= posSH.X())
                    ret = 0;
                else
                    ret = ((posT.X() - posSH.X() - 1) / 4 + 1) * 4;
            }
            else if (posS.X() - posT.X() > 1) {
                ret = ((posS.X() - posT.X() - 1) / 4 + 1) * 4 + ((abs(posS.Y() - posT.Y()) - 1) / 4 + 1) * 4;
            }
            else if (posT.X() - posSH.X() > 1) {
                ret = ((posT.X() - posSH.X() - 1) / 4 + 1) * 4 + ((abs(posS.Y() - posT.Y()) - 1) / 4 + 1) * 4;
            }
            else ret = ((abs(posS.Y() - posT.Y())) / 4 + 1) * 4;
            ret += 0.95;
            return ret * baseCost;
        }
        else if (graph->getVertexType(vertex) == CHANY) {
            COST_T ret = 0;
            auto posS = graph->getPos(vertex), posSH = graph->getPosHigh(vertex);
            auto posT = graph->getPos(target);
            if (posT.X() == posS.X() || posT.X() == posS.X() - 1) {
                if (posT.Y() < posS.Y())
                    ret = ((posS.Y() - posT.Y() - 1) / 4 + 1) * 4;
                else if (posT.Y() <= posSH.Y())
                    ret = 0;
                else
                    ret = ((posT.Y() - posSH.Y() - 1) / 4 + 1) * 4;
            }
            else if (posS.Y() - posT.Y() > 1) {
                ret = ((posS.Y() - posT.Y() - 1) / 4 + 1) * 4 + ((abs(posS.X() - posT.X()) - 1) / 4 + 1) * 4;
            }
            else if (posT.Y() - posSH.Y() > 1) {
                ret = ((posT.Y() - posSH.Y() - 1) / 4 + 1) * 4 + ((abs(posS.X() - posT.X()) - 1) / 4 + 1) * 4;
            }
            else ret = ((abs(posS.X() - posT.X()) - 1) / 4 + 1) * 4;
            ret += 0.95;
            return ret * baseCost;
        }
        else if (graph->getVertexType(vertex) == IPIN) {
            // std::cout << vertex << ' ' << target <<  graph->getPos(vertex).X() << ' ' << graph->getPos(vertex).Y() << ' ' << graph->getPos(target).X() << ' ' << graph->getPos(target).Y() << std::endl;
            if (graph->getPos(vertex).X() >= graph->getPos(target).X() && graph->getPos(vertex).X() <= graph->getPosHigh(target).X()
             && graph->getPos(vertex).Y() >= graph->getPos(target).Y() && graph->getPos(vertex).Y() <= graph->getPosHigh(target).Y()) {
                // if (graph->getVertexCap(vertex) > 0)
                return 0.95 * baseCost;
                // else return 1.0;
            }
            else return 1e9;
        }
        else if (graph->getVertexType(vertex) == GSW) {
            auto posS = graph->getPos(vertex);
            auto posT = graph->getPos(target);
            return (abs(posT.X() - posS.X()) + abs(posT.Y() - posS.Y())) * baseCost;
        }
        return 0;
    }

    bool AStarPathfinder::checkPossiblePathVertex(int vertexId, int source, int sink) {
        if (graph->getVertexByIdx(vertexId)->getGSWConnectLength() > 0) return true;
        auto pos = graph->getPos(vertexId);
        auto posS = graph->getPos(source);
        if (pos.X() == posS.X() && pos.Y() == posS.Y()) return true;
        auto posT = graph->getPos(sink);
        if (pos.X() == posT.X() && pos.Y() == posT.Y()) return true;
        return false;
    }

    COST_T AStarPathfinder::timingDrivenEdgeCost(int vertex, int edgeIdx, COST_T critical) {
        COST_T edgeCost = graph->getEdgeCost(vertex, edgeIdx);
        int nextVertex = graph->getEdge(vertex, edgeIdx);
        COST_T congestCost = edgeCost * graph->getVertexCost(nextVertex);
        return congestCost * (1 - critical) + graph->getEdgeDelay(vertex, edgeIdx) * critical;
    }

    COST_T AStarPathfinder::timingDrivenPridict(int vertex, int target, COST_T critical) {
        int sourceX = graph->getPos(vertex).X(), sourceY = graph->getPos(vertex).Y();
        int sinkX = graph->getPos(target).X(), sinkY = graph->getPos(target).Y();
        int dx = abs(sourceX - sinkX), dy = abs(sourceY - sinkY);
        COST_T costX = (dx / 6) * 120 + ((dx % 6) / 2) * 60 + (dx % 2) * 50;
        COST_T costY = (dy / 6) * 120 + ((dy % 6) / 2) * 60 + (dy % 2) * 50;
        COST_T delayCost = costX + costY;
        if (graph->getVertexType(vertex) == GSW)
        return baseCost * (dx + dy) * (1 - critical) + delayCost * critical;
        return 0;
    }

    RouteStatus AStarPathfinder::routeSingleSink(int targetIdx) {

        int target = net->getSinkByIdx(targetIdx);
        // COST_T critical = net->getSinkCritical(target);
        COST_T critical = 0; //
        // if (iter < 100) critical = net->getSinkCritical(target);


        auto cmp = [](const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) {
            return *a < *b;
        };

        // std::priority_queue<std::shared_ptr<PathNode>, std::vector<std::shared_ptr<PathNode>>, decltype(cmp)> q(cmp);
        BinaryHeap heap;
        heap.init_heap(graph);

        int64_t visitID = 1LL * net->netId * 1e9 + targetIdx * 10 + highFanoutTrick;

        auto addNode = [&](int vertexIdx, int prevIdx, COST_T nodeCost, COST_T predCost, COST_T nodeDelay) {
            if (predCost != 1e9) {
            if (vertexIdx == 38999690)
            if (RouteGraph::debugging) {
                std::cout << "trying to add Node" << vertexIdx << ' ' << graph->getVertexByIdx(vertexIdx)->getName()
                <<" to net " << net->getName()
                 << " cost: " << nodeCost << " predCost: " << predCost
                << " nodeDelay: " << nodeDelay
                << " pos: " << graph->getPos(vertexIdx).X() << ' ' <<
                graph->getPos(vertexIdx).Y() << ' ' << graph->getPosHigh(vertexIdx).X() << ' ' << graph->getPosHigh(vertexIdx).Y() <<
                " check result: " << checkCanExpand(vertexIdx) << std::endl;
                if (prevIdx != -1) std::cout << " prevIdx: " << prevIdx << " " << graph->getVertexByIdx(prevIdx)->getName() << " delay: " << delay[prevIdx] << std::endl;
            }
                if ( checkCanExpand(vertexIdx) && ((visited[vertexIdx] != visitID) || (visited[vertexIdx] == visitID && cost[vertexIdx] > nodeCost))) {
                    // q.push(node);
                    net->inQueueCnt++;
                    visited[vertexIdx] = visitID;
                    cost[vertexIdx] = nodeCost;
                    delay[vertexIdx] = nodeDelay;
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
            for (auto startVertex : startVertices) {
                if (!(checkCanExpand(startVertex) && ((visited[startVertex] != visitID) || (visited[startVertex] == visitID && cost[startVertex] > 0)))) {
                    std::cout << "bug appear!" << ' ' << net->getName() << ' ' << target << ' ' << startVertex << std::endl;
                    getchar();
                }
                if (isTimingDriven)
                    addNode(startVertex, -1, 0, timingDrivenPridict(startVertex, target, critical), delay[startVertex]);
                else
                    addNode(startVertex, -1, 0, predict(startVertex, target), delay[startVertex]);
            }
        }
        else {
            bool flag = false;
            auto posT = graph->getPos(target);
            for (auto startVertex : startVertices) {
                if (!(checkCanExpand(startVertex) && ((visited[startVertex] != visitID) || (visited[startVertex] == visitID && cost[startVertex] > 0)))) {
                    std::cout << "bug appear!" << ' ' << net->getName() << ' ' << target << ' ' << startVertex << std::endl;
                    getchar();
                }
                auto posS = graph->getPos(startVertex);
                if (abs(posS.X() - posT.X()) + abs(posS.Y() - posT.Y()) <= maxHighFanoutAddDist) {

                if (isTimingDriven)
                    addNode(startVertex, -1, 0, timingDrivenPridict(startVertex, target, critical), delay[startVertex]);
                else
                    addNode(startVertex, -1, 0, predict(startVertex, target), delay[startVertex]);
                    flag = true;
                }
            }
            if (!flag) {
                for (auto startVertex : startVertices) {

                if (isTimingDriven)
                    addNode(startVertex, -1, 0, timingDrivenPridict(startVertex, target, critical), delay[startVertex]);
                else
                    addNode(startVertex, -1, 0, predict(startVertex, target), delay[startVertex]);
                }
            }
        }

        while (!heap.is_empty_heap()) {
            auto head = heap.get_heap_head();
            net->outQueueCnt++;

            // if (head->nodeCost != cost[head->headPinId]) continue;
            // std::cout << head->headPinId << ' ' << head->nodeCost << ' ' << head->predCost << std::endl;
            // if (net->getName() == "PARA_25")
            // if (RouteGraph::debugging) {
            // printf("Poping Vertex %d %s\n, prev: %s, Type: %d, cap = %d, cost = %g, predCost = %g, xlow = %d, ylow = %d, xhigh = %d, yhigh = %d\n"
            // , head->index, graph->getVertexByIdx(head->index)->getName().c_str(), prev[head->index] == -1 ? "" : graph->getVertexByIdx(prev[head->index])->getName().c_str(), (int)graph->getVertexType(head->index), graph->getVertexCap(head->index),
            // head->backward_path_cost, head->cost, graph->getPos(head->index).X(),
            // graph->getPos(head->index).Y(), graph->getPosHigh(head->index).X(), graph->getPosHigh(head->index).Y());
            // // getchar();
            // }
            if (head->index == target) {
                // std::cout << "Target " << target << "Found!" << std::endl;
                int now = head->index;
                std::stack<int> pathNodeIds;
                while (now != -1 && (treeNodes[now] == nullptr || treeNodes[now]->net != net)) {
                    // if (net->getName() == "sig_79710")
                    // if (RouteGraph::debugging)
                    //     std::cout << "Node Id: " << now << " pin Name: " << graph->getVertexByIdx(now)->getName() << std::endl;
                    pathNodeIds.push(now);
                    now = prev[now];
                }
                while (!pathNodeIds.empty()) {
                    int node = pathNodeIds.top();
                    pathNodeIds.pop();

                    // std::cout << routetree.getTreeNodeByIdx(prev[node])->nodeId << ' ' << prev[node] <<' ' << node << ' ' << net->getName() << std::endl;
                    // getchar();
                    startVertices.push_back(node);
                    treeNodes[node] = routetree.addNode(treeNodes[prev[node]], node, net, delay[node]);
                }
                heap.free(head);
                return SUCCESS;
            }

            int vertexDegree = graph->getVertexDegree(head->index);
            int vertexIdx = head->index;
            for (int i = 0; i < vertexDegree; i++) {
                int nextVertexIdx = graph->getEdge(vertexIdx, i);
                COST_T edgeCost = graph->getEdgeCost(vertexIdx, i);
                if (isTimingDriven) {
                    addNode(nextVertexIdx, vertexIdx, cost[vertexIdx] + timingDrivenEdgeCost(vertexIdx, i, critical), timingDrivenPridict(nextVertexIdx, target, critical), delay[vertexIdx] + graph->getEdgeDelay(vertexIdx, i));
                    // std::cout << timingDrivenEdgeCost(vertexIdx, i, critical) << ' ' << edgeCost * graph->getVertexCost(nextVertexIdx) << std::endl;
                    // if (graph->getVertexType(nextVertexIdx) == GSW) {
                    //     std::cout << timingDrivenPridict(nextVertexIdx, target, critical) << ' ' << predict(nextVertexIdx, target) << std::endl;
                    //     getchar();
                    // }
                }
                else
                    addNode(nextVertexIdx, vertexIdx, cost[vertexIdx] + edgeCost * graph->getVertexCost(nextVertexIdx), predict(nextVertexIdx, target), delay[vertexIdx] + graph->getEdgeDelay(vertexIdx, i));
            }
            heap.free(head);
        }
        return FAILED;
    }


} // namespace router