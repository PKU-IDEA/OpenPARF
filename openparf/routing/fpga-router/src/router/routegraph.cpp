#include "routegraph.h"
#include "routetree.h"
#include "pathfinder.h"

#include <assert.h>
#include <fstream>
#include <queue>

namespace router {
    COST_T RouteGraph::presFacFirstIter = 0.8;
    COST_T RouteGraph::presFacInit = 1.0;
    COST_T RouteGraph::presFacMult = 10000.0;
    COST_T RouteGraph::presFac;
    COST_T RouteGraph::accFac = 0.8;
    bool RouteGraph::useAStar = true;
    bool RouteGraph::debugging = false;
    bool RouteGraph::dumpingCongestMap = false;

    RouteGraph::RouteGraph(int width, int height, int totalPins) {
        this->width = width;
        this->height = height;

        vertexId.resize(width * height);
        gswVertexId.resize(width * height);

        vertices.reserve(totalPins);
        vertexPos.reserve(totalPins);
        vertexPosHigh.reserve(totalPins);
        vertexCost.reserve(totalPins);
        vertexCap.reserve(totalPins);
        vertexMaxCap.reserve(totalPins);
        vertexInst.reserve(totalPins);
        edges.reserve(totalPins);
        vertexSlack.reserve(totalPins);
        globalGraph = std::make_shared<GlobalRouteGraph> (width, height);
        vertexNum = 0;
        edgeNum = 0;
        GSWVertexNum = 0;
    }

    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, int cap, VertexType vertexType) {
        edges.push_back(std::vector<EdgeNode>());
        // edges[vertexNum].reserve(pin->getConnectSize());
        // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
        for (int i = x_low; i <= x_high; i++)
            for (int j = y_low; j <= y_high; j++)
               vertexId[i * height + j].push_back(vertexNum); 
        vertexNum++;
        // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
        vertexCost.push_back(1);
        vertexPos.push_back(XY<INDEX_T>(x_low, y_low));
        vertexPosHigh.push_back(XY<INDEX_T>(x_high, y_high));
        vertexCap.push_back(cap);
        vertexMaxCap.push_back(cap);
        vertexTypes.push_back(vertexType);
        vertexInst.push_back(-1);
        vertexSlack.push_back(0);
        return vertexNum - 1;
    }


    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin) {
        vertices.push_back(pin);
        edges.push_back(std::vector<EdgeNode>());
        edges[vertexNum].reserve(pin->getConnectSize());
        // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
        vertexId[x_low * height + y_low][pin->getPinId()] = vertexNum; 
        vertexNum++;
        // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
        vertexCost.push_back(1);
        vertexPos.push_back(XY<INDEX_T>(x_low, y_low));
        vertexPosHigh.push_back(XY<INDEX_T>(x_high, y_high));
        vertexCap.push_back(1);
        vertexMaxCap.push_back(1);
        vertexInst.push_back(-1);
        if (pin->getGSWConnectLength()) vertexTypes.push_back(GSW);
        else vertexTypes.push_back(COMMON);
        vertexSlack.push_back(0);
        return vertexNum - 1;
    }

    int RouteGraph::addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Port> port) {
        vertices.push_back(port->getPinByIdx(0));
        edges.push_back(std::vector<EdgeNode>());
        int connSize = 0, width = port->getWidth();
        for (int i = 0; i < width; i++) {
            connSize += port->getPinByIdx(i)->getConnectSize();
            vertexId[x_low * height + y_low][port->getPinByIdx(i)->getPinId()] = vertexNum; 
        }
        edges[vertexNum].reserve(connSize);
        // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
        vertexNum++;
        // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
        vertexCost.push_back(1);
        vertexPos.push_back(XY<INDEX_T>(x_low, y_low));
        vertexPosHigh.push_back(XY<INDEX_T>(x_high, y_high));
        vertexCap.push_back(width);
        vertexMaxCap.push_back(width);
        vertexInst.push_back(-1);
        // std::cout << "vertexNum: " << vertexNum << ' ' << "width: " << width << ' ' << " Cap: " << vertexCap[vertexNum - 1] << std::endl;  
        vertexTypes.push_back(COMMON);
        vertexSlack.push_back(0);
        return vertexNum - 1;
    }

    int RouteGraph::addVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin) {
        vertices.push_back(pin);
        edges.push_back(std::vector<EdgeNode>());
        edges[vertexNum].reserve(pin->getConnectSize());
        // std::cout << vertexId[x * height + y].size() << ' ' << pin->getPinId() << std::endl;
        vertexId[x * height + y][pin->getPinId()] = vertexNum; 
        vertexNum++;
        // std::cout << "RouteGraph::addVertex " << pin->getName() << ' ' << pin << ' ' << x << ' ' << y << ' ' << vertexNum - 1 << std::endl;
        vertexCost.push_back(pin->getPinCost());
        vertexPos.push_back(XY<INDEX_T>(x, y));
        vertexCap.push_back(1);
        vertexMaxCap.push_back(1);
        vertexInst.push_back(-1);
        vertexSlack.push_back(0);
        return vertexNum - 1;
    }

    void RouteGraph::addEdge(int source, int sink, COST_T cost) {
        assert(source >= 0 && source < vertexNum);
        assert(sink >= 0 && sink < vertexNum);
        edges[source].push_back(EdgeNode(sink, cost));
        edgeNum++;
    } 
    void RouteGraph::addEdge(int source, int sink, COST_T cost, COST_T delay) {
        assert(source >= 0 && source < vertexNum);
        assert(sink >= 0 && sink < vertexNum);
        edges[source].push_back(EdgeNode(sink, cost, delay));
        // inputEdges[sink].push_back(EdgeNode(source, cost, delay));
        edgeNum++;
    }

    COST_T RouteGraph::getVertexCost(int vertexIdx) {
        COST_T presCost;
        if (vertexCap[vertexIdx] <= 0) presCost = (1.0 + presFac * (-vertexCap[vertexIdx] + 1));
        else presCost = 1;

        return presCost * vertexCost[vertexIdx];
    }

    int RouteGraph::updateVertexCost() {
        int congestNum = 0;
        int congestNodeCnt[10];
        for (int i = 0; i < 7; i++) congestNodeCnt[i] = 0;
        std::vector<int> congestNums(vertexId.size(), 0);
        for (int i = 0; i < vertexNum; i++) {
            if (vertexCap[i] < 0) {
                vertexCost[i] += (-vertexCap[i]) * accFac;
                congestNum++;
                congestNodeCnt[(int)getVertexType(i)]++;
                congestNums[getPos(i).X() * height + getPos(i).Y()]++;
            } 
        }


        if (dumpingCongestMap) {
            std::ofstream ofs("congestmap.txt");
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++)
                    ofs << congestNums[i * height + j] << ' ';
                ofs << std::endl;
            }
            ofs.close(); 
        }
        if (congestNum && congestNum <= 1000) useAStar = false;
        else useAStar = true;
        std::cout << "congest vertex num: " << congestNum << std::endl;
        return congestNum;
        // for (int i = 1; i < 7; i++) {
        //     std::cout << "Node Type " << i << " Congest Cnt " << congestNodeCnt[i] << std::endl;
        // }
    }

    void RouteGraph::reportStatistics() const {
      std::cout << vertexNum << " vertices, "
        << edgeNum << " edges, "
        << GSWVertexNum << " GSW vertices\n";
      std::cout << "vertices: " << vertices.size() << " entries, " << vertices.capacity() << " caps\n"; 
      std::cout << "vertexPos: " << vertexPos.size() << " entries, " << vertexPos.capacity() << " caps\n"; 
      std::cout << "vertexCost: " << vertexCost.size() << " entries, " << vertexCost.capacity() << " caps\n"; 
      std::size_t num = 0; 
      std::size_t caps = 0; 
      for (auto const& es : edges) {
        num += es.size(); 
        caps += es.capacity();
      }
      std::cout << "edges: " << num << " entries, " << caps << " caps\n"; 
      num = 0; 
      // for (auto const& vs1 : vertexId) {
      //   for (auto const& vs2 : vs1) {
      //     for (std::size_t i = 0; i < vs2.bucket_count(); ++i) {
      //       std::size_t bucket_size = vs2.bucket_size(i); 
      //       if (bucket_size == 0) {
      //         num += 1; 
      //       } else {
      //         num += bucket_size; 
      //       }
      //     }
      //   }
      // }
      // std::cout << "vertexId: " << num << " entries\n"; 
      num = 0; 
      // for (auto const& vs1 : gswVertexId) {
      //   for (auto const& vs2 : vs1) {
      //     for (std::size_t i = 0; i < vs2.bucket_count(); ++i) {
      //       std::size_t bucket_size = vs2.bucket_size(i); 
      //       if (bucket_size == 0) {
      //         num += 1; 
      //       } else {
      //         num += bucket_size; 
      //       }
      //     }
      //   }
      // }
      // std::cout << "gswVertexId: " << num << " entries\n"; 
    }

    LocalRouteGraph::LocalRouteGraph(int vNum) {
        vertexNum = 0;
        edgeNum = 0;
        
        vertexCost.reserve(vNum);
        vertexOriginIdx.reserve(vNum);
        edges.reserve(vNum);
        inputEdges.reserve(vNum);
    }

    int LocalRouteGraph::addVertex(int originIdx, COST_T cost) {
        // std::cout << "Adding " << originIdx << std::endl; 
        vertexOriginIdx.push_back(originIdx);
        vertexCost.push_back(cost);
        edges.push_back(std::vector<EdgeNode>());
        inputEdges.push_back(std::vector<EdgeNode>());
        vertexNum++;
        return vertexNum - 1;
    }

    void LocalRouteGraph::addEdge(int source, int sink, COST_T cost) {
        assert(source >= 0 && source < vertexNum);
        assert(sink >= 0 && sink < vertexNum);
        edges[source].push_back(EdgeNode(sink, cost));
        inputEdges[sink].push_back(EdgeNode(source, cost));
        edgeNum++;
    }


    std::shared_ptr<LocalRouteGraph> RouteGraph::dumpLocalRouteGraph(std::shared_ptr<Net> net) {
        std::unordered_map<int, int> newVertexId;
        auto& guide = net->getGuide();
        if (guide.end_x >= width) 
            guide.end_x = width - 1;
        if (guide.end_y >= height) 
            guide.end_y = height - 1;
        // std::cout << guide.start_x << ' ' << ' ' << guide.start_y << ' ' <<guide.end_x << ' ' << guide.end_y << std::endl;
       // std::set<std::pair<int, int>> visitedPos;

        int sourceX = getPos(net->getSource()).X();
        int sourceY = getPos(net->getSource()).Y();
        // int localVertexNum = 0;
        // localVertexNum += vertexId[sourceX * height + sourceY].size();
        // visitedPos.insert(std::make_pair(sourceX, sourceY));
        // for (int i = 0; i < net->getSinkSize(); i++) {
        //     int pin = net->getSinkByIdx(i);
        //     int x = getPos(pin).X(), y = getPos(pin).Y();
        //     if (visitedPos.find(std::make_pair(x, y)) != visitedPos.end()) continue;
        //     visitedPos.insert(std::make_pair(x, y));
        //     localVertexNum += vertexId[x * height + y].size();
        // }
        // localVertexNum += ((guide.end_x - guide.start_x + 1) * (guide.end_y - guide.start_y + 1) -  visitedPos.size()) * gswVertexId[0].size();

        
        std::set<std::pair<int, int>> visitedPosAddVertex;
        std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph());

        //Add Source Grid Vertex
        for (auto it : vertexId[sourceX * height + sourceY]) {
             if (newVertexId.find(it) == newVertexId.end())
            newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
        }
        visitedPosAddVertex.insert(std::make_pair(sourceX, sourceY));
        //Add Sink Grid Vertex
        for (int i = 0; i < net->getSinkSize(); i++) {
            int pin = net->getSinkByIdx(i);
            int x = getPos(pin).X(), y = getPos(pin).Y();
            if (visitedPosAddVertex.find(std::make_pair(x, y)) != visitedPosAddVertex.end()) continue;
            visitedPosAddVertex.insert(std::make_pair(x, y));
            for (auto it : vertexId[x * height + y]) {
             if (newVertexId.find(it) == newVertexId.end())
                newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
            }
        }

        // Add GSW Vertex
        if (net->useGlobalResult()) {
            auto& gr = net->getGlobalRouteResult();
            for (auto pos : gr) {
                if (visitedPosAddVertex.find(pos) == visitedPosAddVertex.end()) {
                    for (auto it : gswVertexId[pos.first * height + pos.second]) {
                        if (newVertexId.find(it) == newVertexId.end())
                            newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
                        }
                    visitedPosAddVertex.insert(pos);
                }
            }
        }
        else {
            for (int x = guide.start_x; x <= guide.end_x; x++)
                for (int y = guide.start_y; y <= guide.end_y; y++) {
                    // std::cout << x << ' ' << y << std::endl;
                    if (visitedPosAddVertex.find(std::make_pair(x, y)) == visitedPosAddVertex.end())
                    for (auto it : vertexId[x * height + y]) {
                        // std::cout << it << std::endl;
                        if (newVertexId.find(it) == newVertexId.end())
                            newVertexId[it] = localgraph->addVertex(it, vertexCost[it]);
                        }
                }
        }
        //Add Edge
        for (auto it : newVertexId) {
            int degree = getVertexDegree(it.first);
            for (int i = 0; i < degree; i++) {
                int sink = getEdge(it.first, i);
                XY<INDEX_T> pos = getPos(sink);
                XY<INDEX_T> pos_o = getPos(it.first);    
                // if (pos.X() != pos_o.X())
                //     std::cout << pos_o.X() << ' ' << pos_o.Y() << ' ' << getVertexByIdx(it.first)->getName() << "->" << pos.X() << ' ' << pos.Y() << ' ' << getVertexByIdx(sink)->getName() << std::endl;
                    
                if (newVertexId.find(sink) != newVertexId.end()) {
                    localgraph->addEdge(it.second, newVertexId[sink], getEdgeCost(it.first, i));
                }
            }
        }

        // Add RouteTree Node
        std::queue<std::shared_ptr<TreeNode>> q;
        q.push(Pathfinder::routetree.getNetRoot(net));
        while (!q.empty()) {
            std::shared_ptr<TreeNode> node = q.front();
            q.pop();
            // node->localnodeId = newVertexId[node->nodeId];
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                q.push(child);
            }
        }
        return localgraph;
    }
    
}
