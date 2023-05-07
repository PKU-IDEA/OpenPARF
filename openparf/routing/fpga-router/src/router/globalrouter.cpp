#include "globalrouter.h"
#include "path.h"

#include <queue>
#include <algorithm>
#include <chrono>
#include <future>
namespace router {
    GlobalRouter::GlobalRouter(std::shared_ptr<RouteGraph> _graph) {
        graph = _graph;
        // int vertexNum = graph->getVertexNum();
        globalGraph = graph->getGlobalGraph();
        // globalGraph = std::make_shared<GlobalRouteGraph>(graph->getWidth(), graph->getHeight()); 
        // for (int i = 0; i < vertexNum; i++) {
        //     int degree = graph->getVertexDegree(i);
        //     for (int j = 0; j < degree; j++) {
        //         int to = graph->getEdge(i, j);
        //         if (graph->getPos(i).X() != graph->getPos(to).X() || graph->getPos(i).Y() != graph->getPos(to).Y()) {
        //             globalGraph->addEdge(graph->getPos(i).X(), graph->getPos(i).Y(), graph->getPos(to).X(), graph->getPos(to).Y());
        //         }
        //     }
        // }
    }

    void GlobalRouter::run(std::vector<std::shared_ptr<Net>>& netlist) {
        using namespace std::chrono;
        high_resolution_clock::time_point route_s, route_e;
        route_s = high_resolution_clock::now();

        auto cmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> & b) {
            // if (a->getSinkSize() != b->getSinkSize()) return a->getSinkSize() > b->getSinkSize();
            auto guide_a = a->getGuide();
            auto guide_b = b->getGuide();
            return (guide_a.end_x - guide_a.start_x + 1) * (guide_a.end_y - guide_a.start_y + 1)
                <  (guide_b.end_x - guide_b.start_x + 1) * (guide_b.end_y - guide_b.start_y + 1) ;
        };

        std::sort(netlist.begin(), netlist.end(), cmp);

        globalRouteTree.init(globalGraph, graph, netlist);
        globalTreeNodes.resize(globalGraph->getVertexNum());

        int maxIter = 10000;

        for (int iter = 0; iter < maxIter; iter++) {
            if (iter) {
                globalRouteTree.ripup();
                if (globalRouteTree.finish()) break;
            } 
            int successCnt = 0, failedCnt = 0;
            int iNet = 0;
            for (auto net : netlist) {
                RouteStatus status = route(net);
                if (status == SUCCESS) successCnt++;
                else failedCnt++;

                iNet++;
                high_resolution_clock::time_point route_checkpoint = high_resolution_clock::now();
                duration<double, std::ratio<1, 1>> duration_c(route_checkpoint - route_s);
                if (iNet % 10000 == 0)
                    std::cout << "iter #" << iter << " nets #" << iNet << " Runtime: " << duration_c.count() << "s" << std::endl;
            } 
            high_resolution_clock::time_point route_checkpoint = high_resolution_clock::now();
            duration<double, std::ratio<1, 1>> duration_c(route_checkpoint - route_s);
            printf("Iter #%d Total Runtime %lf, Success : Failed = %d : %d\n", iter, duration_c.count(), successCnt, failedCnt);
        }
        globalRouteTree.initNetGlobalResult();
        route_e = high_resolution_clock::now();
        duration<double, std::ratio<1, 1>> duration_s(route_e - route_s);
        printf("Global Route Runtime: %lfs\n", duration_s.count());
        // std::cout << "Global Route SUCCESS : FAILED = " << successCnt << ":" << failedCnt << std::endl;
    }

    RouteStatus GlobalRouter::route(std::shared_ptr<Net> net) {
        // auto sourcePos = graph->getPos(net->getSource());
        // int sourceIdx = globalGraph->getVertexIdx(sourcePos);
        
        int sinkSize = net->getSinkSize();
        std::set<std::pair<int, int>> visitedPos;
        std::vector<int> visitedVertex;
        // visitedPos.insert(std::make_pair(sourcePos.X(), sourcePos.Y()));

        auto root = globalRouteTree.getNetRoot(net);
        std::queue<std::shared_ptr<GlobalTreeNode>> q;
        q.push(root);
        while (!q.empty()) {
            auto node = q.front();
            q.pop(); 
            visitedVertex.push_back(node->nodeId);
            visitedPos.insert(std::make_pair(node->nodeId / graph->getHeight(), node->nodeId % graph->getHeight()));
            globalTreeNodes[node->nodeId] = node;    
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
               q.push(child);
            }
        }

        // visitedVertex.push_back(sourceIdx);
        RouteStatus result = SUCCESS;
        for (int i = 0; i < sinkSize; i++) {
            int sink = net->getSinkByIdx(i);
            int sinkX = graph->getPos(sink).X(), sinkY = graph->getPos(sink).Y();
            if (visitedPos.find(std::make_pair(sinkX, sinkY)) != visitedPos.end()) continue;
            visitedPos.insert(std::make_pair(sinkX, sinkY));
            RouteStatus status = routeSinglePath(net, sinkX * graph->getHeight() + sinkY, visitedVertex, false);
            if (status != SUCCESS) {
                status = routeSinglePath(net, sinkX * graph->getHeight() + sinkY, visitedVertex, true);
                // std::cout << "FUCK" << std::endl;
                result = FAILED;
            }
            if (status != SUCCESS) return FAILED;
        }

        // for (auto vertex : visitedVertex) {
        //     net->addGlobalRouteResult(vertex / graph->getHeight(), vertex % graph->getHeight());
        // }
        // net->useGlobalResult(true);

        return result;
    }

    RouteStatus GlobalRouter::routeSinglePath(std::shared_ptr<Net> net, int sink, std::vector<int>& sources, bool canOverflow) {
        auto cmp = [](const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) {
            return *a < *b;
        };      
        
        int height = graph->getHeight();

        auto calcPredCost = [sink, height](int vertex) {
            int vertexX = vertex / height, vertexY = vertex % height;
            int sinkX = sink / height, sinkY = sink % height;
            return abs(sinkX - vertexX) + abs(sinkY - vertexY);
        };

        std::priority_queue<std::shared_ptr<PathNode>, std::vector<std::shared_ptr<PathNode>>, decltype(cmp)> q(cmp);

        std::vector<COST_T> dis(graph->getWidth() * graph->getHeight(), std::numeric_limits<COST_T>::max());
        
        for (auto source : sources) {
            dis[source] = 0;
            q.push(std::make_shared<PathNode>(source, nullptr, 0, calcPredCost(source)));
        }

        while (!q.empty()) {
            auto now = q.top();
            // std::cout << now->getHeadPin() << std::endl;
            q.pop();

            int nowId = now->getHeadPin();
            if (nowId == sink) {
                std::stack<int> visitedNodes;
                while (now->getPrevNode() != nullptr) {
                    int prevId = now->getPrevNode()->getHeadPin();
                    int currId = now->getHeadPin();
                    // std::cout << prevId << ' ' << currId << std::endl;

                    sources.push_back(currId);
                    int prevDegree = globalGraph->getDegree(prevId);
                    for (int i = 0; i < prevDegree; i++) {
                        if (globalGraph->getEdge(prevId, i).to == now->getHeadPin())
                            visitedNodes.push(i);
                            // globalGraph->decreaseCap(prevId, i);
                    }
                    now = now->getPrevNode();
                }
                std::shared_ptr<GlobalTreeNode> father = globalTreeNodes[now->getHeadPin()];
                while (!visitedNodes.empty()) {
                    int edgeId = visitedNodes.top();
                    visitedNodes.pop();
                    father = globalRouteTree.addNode(father, edgeId, net);
                    globalTreeNodes[father->nodeId] = father;
                }

                return SUCCESS;
            }

            COST_T cost = now->getCost();
            if (dis[nowId] != cost) continue;

            int degree = globalGraph->getDegree(nowId);
            for (int i = 0; i < degree; i++) {
                auto edge = globalGraph->getEdge(nowId, i);
                if (edge.cap <= -2 && !canOverflow) continue;
                // printf("Edge: to=%d cap=%d\n", edge.to, edge.cap);
                if (dis[edge.to] > dis[nowId] + edge.length) {
                    dis[edge.to] = dis[nowId] + edge.length;
                    q.push(std::make_shared<PathNode>(edge.to, now, dis[edge.to], calcPredCost(edge.to)));
                }
            }
        }

        return FAILED;
    }
}