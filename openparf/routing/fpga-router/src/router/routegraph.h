#ifndef ROUTEGRAPH_H
#define ROUTEGRAPH_H

#include "database/module.h"
#include "database/pin.h"
#include "net.h"
#include "utils/utils.h"
#include "globalroutegraph.h"
#include "inst.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <assert.h>
#include <stack>

namespace router {

enum VertexType {
    NONE,
    // For VPR
    IPIN,
    OPIN,
    CHANX,
    CHANY,
    SOURCE,
    SINK,
    // FOR XArch
    GSW,
    COMMON
};

class RouteGraph;
struct EdgeNode {
    INDEX_T to;
    COST_T cost;
    COST_T delay;
    EdgeNode() {}
    EdgeNode(INDEX_T _to, COST_T _cost) : to(_to), cost(_cost), delay(0) {}
    EdgeNode(INDEX_T _to, COST_T _cost, COST_T _delay) : to(_to), cost(_cost), delay(_delay) {}
};

class LocalRouteGraph {
public:
    LocalRouteGraph() : vertexNum(0), edgeNum(0) {}
    LocalRouteGraph(int vNum);

    int addVertex(int originIdx, COST_T cost);
    void addEdge(int source, int sink, COST_T cost);


    int getVertexNum() { return vertexNum; }
    int getEdgeNum() { return edgeNum; }
    int getEdge(int vertexIdx, int edgeIdx) {return edges[vertexIdx][edgeIdx].to; }
    COST_T getEdgeCost(int vertexIdx, int edgeIdx) {return edges[vertexIdx][edgeIdx].cost; }
    int getVertexDegree(int vertexIdx) { return edges[vertexIdx].size(); }
    int getInputDegree(int vertexIdx) { return inputEdges[vertexIdx].size();}
    int getInputEdge(int vertexIdx, int edgeIdx) { return inputEdges[vertexIdx][edgeIdx].to; }
    COST_T getVertexCost(int vertexIdx){ return vertexCost[vertexIdx]; }
    int getOriginIdx(int vertexIdx) {
        if (vertexIdx == -1) return -1;
        return vertexOriginIdx[vertexIdx];
    }
    int getSourceId() { return sourceId; }
    void setSourceId(int id) { sourceId = id; }

private:
    int vertexNum;
    int edgeNum;

    int sourceId;

    std::vector<COST_T> vertexCost;
    std::vector<int> vertexOriginIdx;
    std::vector<std::vector<EdgeNode> > edges;
    std::vector<std::vector<EdgeNode> > inputEdges;

};

class RouteGraph {
public:
    RouteGraph(){}
    RouteGraph(int width, int height, int totalPins);

    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, int cap, VertexType vertexType);
    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin);
    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Port> port);
    int addVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin);
    void addEdge(int source, int sink, COST_T cost);
    void addEdge(int source, int sink, COST_T cost, COST_T delay);

    int getVertexNum() { return vertexNum; }
    int getEdgeNum() { return edgeNum; }

    int getWidth() { return width; }
    int getHeight() { return height; }

    std::shared_ptr<database::Pin> getVertexByIdx(int idx) { return vertices[idx]; }
    int getVertexId(int x, int y, std::shared_ptr<database::Pin> pin) {
        return vertexId[x * height + y][pin->getPinId()];
    }
    int getVertexId(int x, int y, int pinId) {
        return vertexId[x * height + y][pinId];
    }
    VertexType getVertexType(int vertexIdx) { return vertexTypes[vertexIdx]; }
    int getEdge(int vertexIdx, int edgeIdx) {return edges[vertexIdx][edgeIdx].to; }
    COST_T getEdgeCost(int vertexIdx, int edgeIdx) {return edges[vertexIdx][edgeIdx].cost; }
    COST_T getEdgeDelay(int vertexIdx, int edgeIdx) { return edges[vertexIdx][edgeIdx].delay; }
    int getVertexDegree(int vertexIdx) { return edges[vertexIdx].size(); }
    void addVertexCost(int vertexIdx, COST_T addCost = 1.0) { vertexCost[vertexIdx] += addCost; }
    COST_T getVertexCost(int vertexIdx);
    XY<INDEX_T> getPos(int vertexIdx) { return vertexPos[vertexIdx];}
    void setPos(int vertexIdx, INDEX_T x, INDEX_T y) { vertexPos[vertexIdx] = XY<INDEX_T>(x, y); }
    XY<INDEX_T> getPosHigh(int vertexIdx) { return vertexPosHigh[vertexIdx];}
    void addVertexCap(int vertexIdx, int addCap) { vertexCap[vertexIdx] += addCap; }
    int getVertexCap(int vertexIdx) { return vertexCap[vertexIdx]; }
    int getVertexMaxCap(int vertexIdx) { return vertexMaxCap[vertexIdx]; }
    std::vector<std::vector<int>> &vertexIds() { return vertexId; }
    void setVertexInst(int vertexIdx, int instId) { vertexInst[vertexIdx] = instId; }
    int getVertexInst(int vertexIdx) { return vertexInst[vertexIdx]; }
    void setVertexSlack(int vertexIdx, COST_T slack) { vertexSlack[vertexIdx] = slack; }
    COST_T getVertexSlack(int vertexIdx) { return vertexSlack[vertexIdx]; }
    InstList& getInstList() { return instlist; }
    friend class RouteGraphBuilder;

    std::shared_ptr<LocalRouteGraph> dumpLocalRouteGraph(std::shared_ptr<Net> net);

    std::shared_ptr<GlobalRouteGraph> getGlobalGraph() { return globalGraph; }

    int updateVertexCost();

    void reportStatistics() const;

    static COST_T presFacFirstIter;
    static COST_T presFacInit;
    static COST_T presFacMult;
    static COST_T presFac;
    static COST_T accFac;
    static bool   useAStar;
    static bool   debugging;
    static bool   dumpingCongestMap;

    friend class PredictMap;
private:
    int vertexNum;
    int edgeNum;

    int GSWVertexNum;

    int width;
    int height;

    std::vector<std::shared_ptr<database::Pin> > vertices; // 16 * 10^8 ~= 1.6G
    std::vector<XY<INDEX_T>> vertexPos;// 8 * 10^8 ~= 0.8G
    std::vector<XY<INDEX_T>> vertexPosHigh;
    std::vector<COST_T> vertexCost;// 8 * 10^8 ~= 0.8G
    std::vector<int> vertexCap;
    std::vector<int> vertexMaxCap;
    std::vector<std::vector<EdgeNode> > edges; // 24 * 10^8 + 8 * 4 * 1
    std::vector<std::vector<int>> vertexId; // 24 * 10^5 + 4 * 10^8  ~=0 .4G
    std::vector<std::vector<int>> gswVertexId;  // 2G
    std::vector<VertexType> vertexTypes;
    std::vector<int> vertexInst;
    std::vector<COST_T> vertexSlack;

    std::shared_ptr<GlobalRouteGraph> globalGraph;
    InstList instlist;
};

} // namesapce router

#endif // ROUTEGRAPH_H
