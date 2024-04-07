#include "routegraph.h"
#include "routetree.h"
#include "pathfinder.h"

#include <assert.h>
#include <queue>
#include <iostream>

namespace router {
    COST_T RouteGraph::presFacFirstIter = 4.0;
    COST_T RouteGraph::presFacInit = 16.0;
    COST_T RouteGraph::presFacMult = 4.0;
    COST_T RouteGraph::presFac;
    COST_T RouteGraph::accFac = 1.0;
    bool RouteGraph::useAStar = true;
    bool RouteGraph::debugging = false;

    int  RouteGraph::boundingBoxExtensionX = 3;
    int  RouteGraph::boundingBoxExtensionY = 15;
    bool RouteGraph::enlargeBoundingBox = false;
    int  RouteGraph::extensionYIncrement = 2;
    int  RouteGraph::extensionXIncrement = 1;
    COST_T   RouteGraph::wirelengthWeight = 0.8;
    COST_T   RouteGraph::timingWeight = 0.35;
    COST_T   RouteGraph::timingMultiplier = 1.0;
    COST_T   RouteGraph::shareExponent = 2.0;
    COST_T   RouteGraph::criticalityExponent = 3.0;
    COST_T   RouteGraph::minRerouteCriticality = 0.85;
    int      RouteGraph::reroutePercentage = 3;
    COST_T   RouteGraph::historicalCongestionFactor = 1.0;
    COST_T   RouteGraph::presentCongestionMultiplier = 2.0;
    COST_T   RouteGraph::initialPresentCongestionFactor = 0.5;
    // COST_T   RouteGraph::initialPresentCongestionFactor = 0.8;

    COST_T   RouteGraph::maxPresentCongestionFactor = std::numeric_limits<float>::max();


    RouteGraph::RouteGraph(int width, int height, int totalPins) {
        this->width = width;
        this->height = height;

        // vertexId.resize(width * height);
        // gswVertexId.resize(width * height);
        // clkVertexId.resize(width * height);

        // vertices.reserve(totalPins);
        // vertexPos.reserve(totalPins);
        // vertexPosHigh.reserve(totalPins);
        // vertexPosLow.reserve(totalPins);
        // vertexCost.reserve(totalPins);
        // vertexCap.reserve(totalPins);
        // vertexMaxCap.reserve(totalPins);
        // edges.reserve(totalPins);
        vertices.reserve(30000000);
        edges.reserve(135000000);
        vertexNum = 0;
        edgeNum = 0;
        GSWVertexNum = 0;
        // gridVertexNum = 0;
    }

    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, int cap, VertexType vertexType) {
        int id = vertexNum;
        vertexNum++;
        vertices.emplace_back();
        vertices[id].id = id;
        vertices[id].cap = cap;
        vertices[id].posLow = XY<INDEX_T>(x_low, y_low);
        vertices[id].posHigh = XY<INDEX_T>(x_high, y_high);
        vertices[id].cost = 1.0;
        vertices[id].histCost = 1;
        vertices[id].type = vertexType;
        return id;
    }


    // int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin) {
    //     vertices.push_back(pin);
    //     edges.push_back(std::vector<EdgeNode>());
    //     edges[vertexNum].reserve(pin->getConnectSize());
    //     // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
    //     vertexId[x_low * height + y_low][pin->getPinId()] = vertexNum; 
    //     vertexNum++;
    //     // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
    //     vertexCost.push_back(1);
    //     vertexPos.push_back(XY<INDEX_T>(x_low, y_low));
    //     vertexPosHigh.push_back(XY<INDEX_T>(x_high, y_high));
    //     vertexPosLow.push_back(XY<INDEX_T>(x_low, y_low));
    //     vertexCap.push_back(1);
    //     vertexMaxCap.push_back(1);
    //     if (pin->getGSWConnectLength()) vertexTypes.push_back(GSW);
    //     else if (pin->getGSWO()) vertexTypes.push_back(GSWO);
    //     else vertexTypes.push_back(COMMON);

    //     // if (x_low == 0 && y_low == 0) {
    //     //     // if (pin->getGSWConnectPin() != "") {
    //     //         std::cout << "pin name " << pin->getName() << ", gswConnectLength " << pin->getGSWConnectLength()
    //     //             << ", connectSize " << pin->getConnectSize() << ", GSWConnectPin " << pin->getGSWConnectPin()
    //     //             << ", GSWO " << pin->getGSWO()
    //     //             << std::endl;
    //     //     // }
    //     // }

    //     return vertexNum - 1;
    // }

    
    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap) {
        int id = vertexNum;
        vertexNum++;
        vertices.emplace_back();
        vertices[id].id = id;
        vertices[id].cap = cap;
        vertices[id].posLow = XY<INDEX_T>(x_low, y_low);
        vertices[id].posHigh = XY<INDEX_T>(x_high, y_high);
        vertices[id].cost = 1.0;
        vertices[id].histCost = 1;
        vertices[id].type = COMMON;
        vertices[id].pin = pin;
        return id;
        // return vertexNum - 1;
    }

    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap, IntentCode nodeType) {
        int id = vertexNum;
        vertexNum++;
        vertices.emplace_back();
        vertices[id].id = id;
        vertices[id].cap = cap;
        vertices[id].posLow = XY<INDEX_T>(x_low, y_low);
        vertices[id].posHigh = XY<INDEX_T>(x_high, y_high);
        vertices[id].cost = 1.0;
        vertices[id].histCost = 1;
        vertices[id].type = COMMON;
        vertices[id].pin = pin;
        vertices[id].ic = nodeType;
        vertices[id].baseCost = baseCost;
        return id;
    };

    int RouteGraph::addVertex(int nodeIdx, INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap, IntentCode nodeType){
        auto& v = vertices[nodeIdx];
        v.id = nodeIdx;
        v.cap = cap;
        v.posLow = XY<INDEX_T>(x_low, y_low);
        v.posHigh = XY<INDEX_T>(x_high, y_high);
        v.cost = 1.0;
        v.histCost = 1;
        v.type = COMMON;
        v.pin = pin;
        v.ic = nodeType;
        v.baseCost = baseCost;
        return nodeIdx;
    }

    // int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Port> port) {
    //     vertices.push_back(port->getPinByIdx(0));
    //     edges.push_back(std::vector<EdgeNode>());
    //     int connSize = 0, width = port->getWidth();
    //     for (int i = 0; i < width; i++) {
    //         connSize += port->getPinByIdx(i)->getConnectSize();
    //         vertexId[x_low * height + y_low][port->getPinByIdx(i)->getPinId()] = vertexNum; 
    //     }
    //     edges[vertexNum].reserve(connSize);
    //     // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
    //     vertexNum++;
    //     // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
    //     vertexCost.push_back(1);
    //     vertexPos.push_back(XY<INDEX_T>(x_low, y_low));
    //     vertexPosHigh.push_back(XY<INDEX_T>(x_high, y_high));
    //     vertexPosLow.push_back(XY<INDEX_T>(x_low, y_low));
    //     vertexCap.push_back(width);
    //     vertexMaxCap.push_back(width);
    //     // std::cout << "vertexNum: " << vertexNum << ' ' << "width: " << width << ' ' << " Cap: " << vertexCap[vertexNum - 1] << std::endl;  
    //     vertexTypes.push_back(COMMON);
    //     gridVertexNum++;
    //     return vertexNum - 1;
    // }

    // int RouteGraph::addVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin) {
    //     vertices.push_back(pin);
    //     edges.push_back(std::vector<EdgeNode>());
    //     edges[vertexNum].reserve(pin->getConnectSize());
    //     // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
    //     vertexId[x * height + y][pin->getPinId()] = vertexNum; 
    //     vertexNum++;
    //     // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
    //     vertexCost.push_back(pin->getPinCost());
    //     vertexPos.push_back(XY<INDEX_T>(x, y));
    //     vertexCap.push_back(1);
    //     vertexMaxCap.push_back(1);
    //     gridVertexNum++;
    //     return vertexNum - 1;
    // }

    // int RouteGraph::addClkVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin) {
    //     vertices.push_back(pin);
    //     edges.push_back(std::vector<EdgeNode>());
    //     edges[vertexNum].reserve(pin->getConnectSize());
    //     clkVertexId[x * height + y][pin->getPinId()] = vertexNum; 
    //     vertexNum++;
    //     // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
    //     vertexCost.push_back(pin->getPinCost());
    //     vertexPos.push_back(XY<INDEX_T>(x, y));
    //     vertexPosHigh.push_back(XY<INDEX_T>(x, y));
    //     vertexPosLow.push_back(XY<INDEX_T>(x, y));
    //     vertexCap.push_back(1);
    //     vertexMaxCap.push_back(1);
    //     vertexTypes.push_back(COMMON);
    //     gridVertexNum++;
    //     return vertexNum - 1;
    // }

    void RouteGraph::addEdge(int source, int sink, COST_T cost) {
        // assert(source >= 0 && source < vertexNum);
        // assert(sink >= 0 && sink < vertexNum);
        // edges[source].push_back(EdgeNode(sink, cost));
        // edgeNum++;
        // if (source == 27617388) {
        //     std::cout << sink << ' ' << cost << ' ' << vertices[sink].pin->getName() << std::endl;
        // }
        int id = edges.size();
        edges.emplace_back();
        edgeNum++;
        edges[id].id = id;
        edges[id].from = source;
        edges[id].to = sink;
        edges[id].preFromEdge = vertices[source].headOutEdgeId;
        vertices[source].headOutEdgeId = id;
        vertices[source].outDegree++;
        edges[id].preToEdge = vertices[sink].headInEdgeId;
        vertices[sink].headInEdgeId = id;
        vertices[sink].inDegree++;
        edges[id].cost = cost;
    } 

    void RouteGraph::addEdge(int id, int source, int sink, COST_T cost){
        edges[id].id = id;
        edges[id].from = source;
        edges[id].to = sink;
        edges[id].cost = cost;
    }

    void RouteGraph::edgeOut(int id){
        int source = edges[id].from;
        int sink = edges[id].to;
        edges[id].preFromEdge = vertices[source].headOutEdgeId;
        vertices[source].headOutEdgeId = id;
        vertices[source].outDegree++;
    }

    void RouteGraph::edgeIn(int id){
        int source = edges[id].from;
        int sink = edges[id].to;
        edges[id].preToEdge = vertices[sink].headInEdgeId;
        vertices[sink].headInEdgeId = id;
        vertices[sink].inDegree++;
    }

    COST_T RouteGraph::getVertexCost(int vertexIdx) {
        COST_T presentCongestionCost;
        if (vertices[vertexIdx].cap <= vertices[vertexIdx].occ)
            presentCongestionCost = (1.0 + vertices[vertexIdx].cost * (vertices[vertexIdx].occ - vertices[vertexIdx].cap + 1));
        else
            presentCongestionCost = 1;
        // return presentCongestionCost * vertices[vertexIdx].cost;

        // biasCost = rnode.getBaseCost() / net.getConnections().size() *
                    // (Math.abs(rnode.getEndTileXCoordinate() - net.getXCenter()) + Math.abs(rnode.getEndTileYCoordinate() - net.getYCenter())) / net.getDoubleHpwl();
        // rnode.getBaseCost() * rnode.getHistoricalCongestionCost() * presentCongestionCost / sharingFactor

        // std::cout << "--> baseCost: " << vertices[vertexIdx].baseCost
        //         << " histCost: " << vertices[vertexIdx].histCost
        //         << " presentCongestionCost: " << presentCongestionCost << std::endl;

        return vertices[vertexIdx].baseCost * vertices[vertexIdx].histCost * presentCongestionCost;
    }

    // private float getNodeCost(RouteNode rnode, Connection connection, int countSameSourceUsers, float sharingFactor) {
    //     boolean hasSameSourceUsers = countSameSourceUsers!= 0;
    //     float presentCongestionCost;

    //     if (hasSameSourceUsers) {// the rnode is used by other connection(s) from the same net
    //         int overoccupancy = rnode.getOccupancy() - RouteNode.capacity;
    //         // make the congestion cost less for the current connection
    //         presentCongestionCost = 1 + overoccupancy * presentCongestionFactor;
    //     } else {
    //         presentCongestionCost = rnode.getPresentCongestionCost();
    //     }

    //     float biasCost = 0;
    //     if (!rnode.isTarget() && rnode.getType() != RouteNodeType.SUPER_LONG_LINE) {
    //         NetWrapper net = connection.getNetWrapper();
    //         biasCost = rnode.getBaseCost() / net.getConnections().size() *
    //                 (Math.abs(rnode.getEndTileXCoordinate() - net.getXCenter()) + Math.abs(rnode.getEndTileYCoordinate() - net.getYCenter())) / net.getDoubleHpwl();
    //     }

    //     return rnode.getBaseCost() * rnode.getHistoricalCongestionCost() * presentCongestionCost / sharingFactor + biasCost;
    // }


    void RouteGraph::updateVertexCost() {
        // int congestNum = 0;
        // for (int i = 0; i < vertexNum; i++) {
        //     if (vertices[i].cap < vertices[i].occ) {
        //         vertices[i].cost += (vertices[i].occ - vertices[i].cap) * accFac;
        //         congestNum++;
        //     } 
        // }
        // if (congestNum && congestNum <= 1000)
        //     useAStar = false;
        // else
        //     useAStar = true;
        int congestNum = 0;
        for (int i = 0; i < vertexNum; i++) {
            // if (vertices[i].ic == NODE_CLE_OUTPUT) {
            //     vertices[i].cost = 0;
            //     vertices[i].histCost = 0;
            // }

            int overuse = vertices[i].occ - vertices[i].cap;
            if (overuse == 0) {
                vertices[i].cost = 1 + RouteGraph::presFac;
            } else if (overuse > 0) {
                vertices[i].cost = 1 + RouteGraph::presFac + overuse * RouteGraph::presFac;
                vertices[i].histCost += overuse * RouteGraph::historicalCongestionFactor;
                congestNum++;
                // std::cout << "vertexId: " << i 
                //     << " nodeType: " << vertices[i].ic
                //     << " cost: " << vertices[i].cost << " histCost: " << vertices[i].histCost
                //     << " presFac: " << RouteGraph::presFac << std::endl;
            } else {
                assert(overuse < 0);
                assert(vertices[i].cost == 1);
            }
        }
        useAStar = !(congestNum && congestNum <= 1000);
        // useAStar = false;
        std::cout << "congest vertex num: " << congestNum << std::endl;
    }

    // /**
    //  * Updates present congestion cost and historical congestion cost of rnodes.
    //  */
    // private void updateCost() {
    //     overUsedRnodes.clear();
    //     for (RouteNode rnode : routingGraph.getRnodes()) {
    //         int overuse=rnode.getOccupancy() - RouteNode.capacity;
    //         if (overuse == 0) {
    //             rnode.setPresentCongestionCost(1 + presentCongestionFactor);
    //         } else if (overuse > 0) {
    //             overUsedRnodes.add(rnode);
    //             rnode.setPresentCongestionCost(1 + (overuse + 1) * presentCongestionFactor);
    //             rnode.setHistoricalCongestionCost(rnode.getHistoricalCongestionCost() + overuse * historicalCongestionFactor);
    //         } else {
    //             assert(overuse < 0);
    //             assert(rnode.getPresentCongestionCost() == 1);
    //         }
    //     }
    // }










    // void RouteGraph::reportStatistics() const {
    //   std::cout << vertexNum << " vertices, "
    //     << edgeNum << " edges, "
    //     << GSWVertexNum << " GSW vertices\n";
    //   std::cout << "vertices: " << vertices.size() << " entries, " << vertices.capacity() << " caps\n"; 
    //   std::cout << "vertexPos: " << vertexPos.size() << " entries, " << vertexPos.capacity() << " caps\n"; 
    //   std::cout << "vertexCost: " << vertexCost.size() << " entries, " << vertexCost.capacity() << " caps\n"; 
    //   std::size_t num = 0; 
    //   std::size_t caps = 0; 
    //   for (auto const& es : edges) {
    //     num += es.size(); 
    //     caps += es.capacity();
    //   }
    //   std::cout << "edges: " << num << " entries, " << caps << " caps\n"; 
    //   num = 0; 
    //   // for (auto const& vs1 : vertexId) {
    //   //   for (auto const& vs2 : vs1) {
    //   //     for (std::size_t i = 0; i < vs2.bucket_count(); ++i) {
    //   //       std::size_t bucket_size = vs2.bucket_size(i); 
    //   //       if (bucket_size == 0) {
    //   //         num += 1; 
    //   //       } else {
    //   //         num += bucket_size; 
    //   //       }
    //   //     }
    //   //   }
    //   // }
    //   // std::cout << "vertexId: " << num << " entries\n"; 
    //   num = 0; 
    //   // for (auto const& vs1 : gswVertexId) {
    //   //   for (auto const& vs2 : vs1) {
    //   //     for (std::size_t i = 0; i < vs2.bucket_count(); ++i) {
    //   //       std::size_t bucket_size = vs2.bucket_size(i); 
    //   //       if (bucket_size == 0) {
    //   //         num += 1; 
    //   //       } else {
    //   //         num += bucket_size; 
    //   //       }
    //   //     }
    //   //   }
    //   // }
    //   // std::cout << "gswVertexId: " << num << " entries\n"; 
    // }

   

    // std::shared_ptr<LocalRouteGraph> RouteGraph::dumpLocalRouteGraph(std::shared_ptr<Net> net) {
    //     std::unordered_map<int, int> newVertexId;
    //     auto& guide = net->getGuide();
    //     if (guide.end_x >= width) 
    //         guide.end_x = width - 1;
    //     if (guide.end_y >= height) 
    //         guide.end_y = height - 1;
    //     // std::cout << guide.start_x << ' ' << ' ' << guide.start_y << ' ' <<guide.end_x << ' ' << guide.end_y << std::endl;
    //    // std::set<std::pair<int, int>> visitedPos;

    //     int sourceX = getPos(net->getSource()).X();
    //     int sourceY = getPos(net->getSource()).Y();
    //     // int localVertexNum = 0;
    //     // localVertexNum += vertexId[sourceX * height + sourceY].size();
    //     // visitedPos.insert(std::make_pair(sourceX, sourceY));
    //     // for (int i = 0; i < net->getSinkSize(); i++) {
    //     //     int pin = net->getSinkByIdx(i);
    //     //     int x = getPos(pin).X(), y = getPos(pin).Y();
    //     //     if (visitedPos.find(std::make_pair(x, y)) != visitedPos.end()) continue;
    //     //     visitedPos.insert(std::make_pair(x, y));
    //     //     localVertexNum += vertexId[x * height + y].size();
    //     // }
    //     // localVertexNum += ((guide.end_x - guide.start_x + 1) * (guide.end_y - guide.start_y + 1) -  visitedPos.size()) * gswVertexId[0].size();

        
    //     std::set<std::pair<int, int>> visitedPosAddVertex;
    //     std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph(vertexNum));

    //     //Add Source Grid Vertex
    //     for (auto it : vertexId[sourceX * height + sourceY]) {
    //          if (newVertexId.find(it) == newVertexId.end())
    //         newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
    //     }
    //     visitedPosAddVertex.insert(std::make_pair(sourceX, sourceY));
    //     //Add Sink Grid Vertex
    //     for (int i = 0; i < net->getSinkSize(); i++) {
    //         int pin = net->getSinkByIdx(i);
    //         int x = getPos(pin).X(), y = getPos(pin).Y();
    //         if (visitedPosAddVertex.find(std::make_pair(x, y)) != visitedPosAddVertex.end()) continue;
    //         visitedPosAddVertex.insert(std::make_pair(x, y));
    //         for (auto it : vertexId[x * height + y]) {
    //          if (newVertexId.find(it) == newVertexId.end())
    //             newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
    //         }
    //     }

    //     // Add GSW Vertex
    //     if (net->useGlobalResult()) {
    //         auto& gr = net->getGlobalRouteResult();
    //         for (auto pos : gr) {
    //             if (visitedPosAddVertex.find(pos) == visitedPosAddVertex.end()) {
    //                 for (auto it : gswVertexId[pos.first * height + pos.second]) {
    //                     if (newVertexId.find(it) == newVertexId.end())
    //                         newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
    //                     }
    //                 visitedPosAddVertex.insert(pos);
    //             }
    //         }
    //     }
    //     else {
    //         for (int x = guide.start_x; x <= guide.end_x; x++)
    //             for (int y = guide.start_y; y <= guide.end_y; y++) {
    //                 // std::cout << x << ' ' << y << std::endl;
    //                 if (visitedPosAddVertex.find(std::make_pair(x, y)) == visitedPosAddVertex.end())
    //                 for (auto it : vertexId[x * height + y]) {
    //                     // std::cout << it << std::endl;
    //                     if (newVertexId.find(it) == newVertexId.end())
    //                         newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
    //                     }
    //             }
    //     }
    //     //Add Edge
    //     for (auto it : newVertexId) {
    //         int degree = getVertexDegree(it.first);
    //         for (int i = 0; i < degree; i++) {
    //             int sink = getEdge(it.first, i);
    //             XY<INDEX_T> pos = getPos(sink);
    //             XY<INDEX_T> pos_o = getPos(it.first);    
    //             // if (pos.X() != pos_o.X())
    //             //     std::cout << pos_o.X() << ' ' << pos_o.Y() << ' ' << getVertexByIdx(it.first)->getName() << "->" << pos.X() << ' ' << pos.Y() << ' ' << getVertexByIdx(sink)->getName() << std::endl;
                    
    //             if (newVertexId.find(sink) != newVertexId.end()) {
    //                 localgraph->addEdge(it.second, newVertexId[sink], getEdgeCost(it.first, i));
    //             }
    //         }
    //     }

    //     // Add RouteTree Node
    //     std::queue<std::shared_ptr<TreeNode>> q;
    //     q.push(Pathfinder::routetree.getNetRoot(net));
    //     while (!q.empty()) {
    //         std::shared_ptr<TreeNode> node = q.front();
    //         q.pop();
    //         // node->localnodeId = newVertexId[node->nodeId];
    //         for (auto child = node->firstChild; child != nullptr; child = child->right) {
    //             q.push(child);
    //         }
    //     }
    //     return localgraph;
    // }

    void RouteGraph::checkVertexConnect(std::string const& checkFile) {
        std::ifstream ifs(checkFile);
        std::string s;
        int v1, v2;
        while (ifs >> s >> v1 >> v2) {
            std::cout << "---------------------------" << std::endl;
            std::cout << s << ' ' << v1 << ' ' << v2 << std::endl;
            std::cout << "---------------------------" << std::endl;
            for (int i = 0; i < vertexNum; i++) {
                if (vertices[i].pin->getName() == s && vertices[i].posLow.X() == v1 && vertices[i].posLow.Y() == v2) {
                    checkSet.insert(i);
                    // for (int j = 0; j < edges[i].size(); j++) {
                    //     int nex = edges[i][j].to;
                    //     std::cout << nex << ' ' << vertices[nex]->getName() << ' ' << vertexPos[nex].X() << ' ' << vertexPos[nex].Y() << std::endl;
                    // }
                    for (int edgeId = vertices[i].headOutEdgeId; edgeId != -1; edgeId = edges[edgeId].preFromEdge) {
                        std::cout << edges[edgeId].to << ' ' << vertices[edges[edgeId].to].pin->getName() << ' ' << vertices[edges[edgeId].to].posLow.X() << vertices[edges[edgeId].to].posLow.Y(); 
                    }
                    std::cout << "---------------------------" << std::endl;
                    std::cout << std::endl;
                    break;
                }
            }
        }
        ifs.close();
    }
    
}
