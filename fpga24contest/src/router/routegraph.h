#ifndef ROUTEGRAPH_H
#define ROUTEGRAPH_H

// #include "../database/module.h"
#include "../database/pin.h"
#include "net.h"
#include "../utils/utils.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <assert.h>
#include <stack>

namespace router {

enum VertexType {
    NONE,
    COMMON,
    // FOR FPGAIF
    NETSINK,
    INTTILE,
    SOURCE
};

enum IntentCode {
    // UltraScale
    INTENT_DEFAULT,
    NODE_OUTPUT,
    NODE_DEDICATED,
    NODE_GLOBAL_VDISTR,
    NODE_GLOBAL_HROUTE,
    NODE_GLOBAL_HDISTR,
    NODE_PINFEED,
    NODE_PINBOUNCE,
    NODE_LOCAL,
    NODE_HLONG,
    NODE_SINGLE,
    NODE_DOUBLE,
    NODE_HQUAD,
    NODE_VLONG,
    NODE_VQUAD,
    NODE_OPTDELAY,
    NODE_GLOBAL_VROUTE,
    NODE_GLOBAL_LEAF,
    NODE_GLOBAL_BUFG,

    //UltraScale+
    NODE_LAGUNA_DATA,
    NODE_CLE_OUTPUT,
    NODE_INT_INTERFACE,
    NODE_LAGUNA_OUTPUT
};

class RouteGraph;
struct EdgeNode {
    INDEX_T to;
    COST_T cost;
    EdgeNode() {}
    EdgeNode(INDEX_T _to, COST_T _cost) : to(_to), cost(_cost) {}
};

struct RouteGraphVertex {
    int id;
    XY<INDEX_T> posLow;
    XY<INDEX_T> posHigh;
    COST_T baseCost = 0;
    COST_T cost = 0;
    COST_T length = 0;
    int cap = 0;
    int occ = 0;
    COST_T histCost = 1.0;
    VertexType type;
    IntentCode ic;
    std::shared_ptr<database::Pin> pin;
    int pinNodeName;
    int tileRow;
    int tileCol;
    int inDegree = 0;
    int outDegree = 0;
    int headOutEdgeId = -1;
    int headInEdgeId = -1;
};

struct RouteGraphEdge {
    int id;
    int from;
    int to;
    COST_T cost;
    int preFromEdge;
    int preToEdge;
};



class RouteGraph {
public:
    RouteGraph(){}
    RouteGraph(int width, int height, int totalPins);

    void setVertexNum(int v){
        vertices.resize(v);
        vertexNum = v;
    }

    void setEdgeNum(int e){
        edgeNum = e;
        edges.resize(e);
    }

    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, int cap, VertexType vertexType);
    // int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin);
    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap);
    int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap, IntentCode nodeType);
    int addVertex(int nodeIdx, INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap, IntentCode nodeType);
    // int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Pin> pin, int cap, uint32_t nodeType, COST_T length);
    // int addVertex(INDEX_T x_low, INDEX_T y_low, INDEX_T x_high, INDEX_T y_high, std::shared_ptr<database::Port> port);
    // int addVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin);
    void addEdge(int source, int sink, COST_T cost);

    void addEdge(int id, int source, int sink, COST_T cost);
    void edgeOut(int id);
    void edgeIn(int id);

    // int addClkVertex(INDEX_T x, INDEX_T y, std::shared_ptr<database::Pin> pin); 

    int getVertexNum() { return vertexNum; }
    int getEdgeNum() { return edgeNum; }

    int getWidth() { return width; }
    int getHeight() { return height; }

    std::shared_ptr<database::Pin> getVertexByIdx(int idx) { return vertices[idx].pin; }
    // int getVertexId(int x, int y, std::shared_ptr<database::Pin> pin) { 
    //     return vertexId[x * height + y][pin->getPinId()]; 
    // }
    // int getVertexId(int x, int y, int pinId) { 
    //     return vertexId[x * height + y][pinId]; 
    // }

    
    // int getClkVertexId(int x, int y, std::shared_ptr<database::Pin> pin) { 
    //     return clkVertexId[x * height + y][pin->getPinId()]; 
    // }
    // int getClkVertexId(int x, int y, int pinId) { 
    //     return clkVertexId[x * height + y][pinId]; 
    // }
    VertexType getVertexType(int vertexIdx) { return vertices[vertexIdx].type; }
    void setVertexType(int vertexIdx, VertexType type) { vertices[vertexIdx].type = type; }

    IntentCode getVertexIC(int vertexIdx) { return vertices[vertexIdx].ic; }

    RouteGraphEdge& getEdge(int edgeIdx) {return edges[edgeIdx]; }
    // COST_T getEdgeCost(int vertexIdx, int edgeIdx) {return edges[vertexIdx][edgeIdx].cost; }

    void setVertexLengthAndBaseCost(int vertexIdx, int length) {
        vertices[vertexIdx].length = length;
        vertices[vertexIdx].baseCost = baseCost;
        switch (vertices[vertexIdx].ic) {
        // NOTE: IntentCode is device-dependent
            case NODE_OUTPUT:       // LUT route-thru
            case NODE_LOCAL:
            case INTENT_DEFAULT:
                // assert(length <= 1);
                // break;
            case NODE_SINGLE:
                // assert(length <= 2);
                if (length == 2) vertices[vertexIdx].baseCost *= length;
                // break;
            case NODE_DOUBLE:
                // assert(length <= 2);
                // // Typically, length = 1 (since tile X is not equal)
                // // In US, have seen length = 2, e.g. VU440's INT_X171Y827/EE2_E_BEG7.
                if (length == 2) vertices[vertexIdx].baseCost *= length;
                // break;
            case NODE_HQUAD:
                // assert (length != 0 || node.getAllDownhillNodes().isEmpty());
                vertices[vertexIdx].baseCost = 0.35f * length;
                // break;
            case NODE_VQUAD:
                // In case of U-turn nodes
                if (length != 0) vertices[vertexIdx].baseCost = 0.15f * length;// VQUADs have length 4 and 5
                // break;
            case NODE_HLONG:
                // assert (length != 0 || node.getAllDownhillNodes().isEmpty());
                vertices[vertexIdx].baseCost = 0.15f * length;// HLONGs have length 6 and 7
                // break;
            case NODE_VLONG:
                vertices[vertexIdx].baseCost = 0.7f;
                // break;
            default:
                vertices[vertexIdx].baseCost = baseCost;
        }
    }

    void setVertexLength(int vertexIdx, COST_T length) { vertices[vertexIdx].length = length; }
    COST_T getVertexLength(int vertexIdx) { return vertices[vertexIdx].length; }

    int getVertexDegree(int vertexIdx) { return vertices[vertexIdx].outDegree; }
    int getVertexInDegree(int vertexIdx) { return vertices[vertexIdx].inDegree; }

    int getHeadOutEdgeIdx(int vertexIdx) { return vertices[vertexIdx].headOutEdgeId; }
    int getHeadInEdgeIdx(int vertexIdx) { return vertices[vertexIdx].headInEdgeId; }

    void addVertexCost(int vertexIdx, COST_T addCost = 1.0) { vertices[vertexIdx].histCost += addCost; }
    COST_T getVertexCost(int vertexIdx); 
    XY<INDEX_T> getPos(int vertexIdx) { return vertices[vertexIdx].posLow;}
    void setPos(int vertexIdx, INDEX_T x, INDEX_T y) { vertices[vertexIdx].posLow = XY<INDEX_T>(x, y); }
    XY<INDEX_T> getPosHigh(int vertexIdx) { return vertices[vertexIdx].posHigh;}
    XY<INDEX_T> getPosLow(int vertexIdx) { return vertices[vertexIdx].posLow;}
    void addVertexCap(int vertexIdx, int addCap) { vertices[vertexIdx].occ -= addCap; }
    int getVertexCap(int vertexIdx) { return vertices[vertexIdx].cap - vertices[vertexIdx].occ; }
    int getVertexMaxCap(int vertexIdx) { return vertices[vertexIdx].cap; }
    int isVertexNotOccupied(int vertexIdx) { return vertices[vertexIdx].occ == 0; }
    // std::vector<std::vector<int>> &vertexIds() { return vertexId; }
    // std::vector<std::vector<int>> &gswVertexIds() { return gswVertexId; }
    friend class RouteGraphBuilder;

    // std::shared_ptr<LocalRouteGraph> dumpLocalRouteGraph(std::shared_ptr<Net> net);


    void updateVertexCost();

    void checkVertexConnect(std::string const& checkFile);

    // void reportStatistics() const; 

    static COST_T presFacFirstIter;
    static COST_T presFacInit;
    static COST_T presFacMult;
    static COST_T presFac;
    static COST_T accFac;
    static bool   useAStar;
    static bool   debugging;

    static int  boundingBoxExtensionX;
    static int  boundingBoxExtensionY;
    static bool enlargeBoundingBox;
    static int  extensionYIncrement;
    static int  extensionXIncrement;
    static COST_T   wirelengthWeight;
    static COST_T   timingWeight;
    static COST_T   timingMultiplier;
    static COST_T   shareExponent;
    static COST_T   criticalityExponent;
    static COST_T   minRerouteCriticality;
    static int      reroutePercentage;
    static COST_T   historicalCongestionFactor;
    static COST_T   presentCongestionMultiplier;
    static COST_T   initialPresentCongestionFactor;

    static COST_T   maxPresentCongestionFactor;


    friend class PredictMap;

    std::set<int> checkSet;

    auto& getVertices() {return vertices;}
    auto& getEdges() {return edges;}
    void setWidth(int w) {width = w;}
    void setHeight(int h) {height = h;}
private:
    int vertexNum;
    int edgeNum;

    int GSWVertexNum;

    int width;
    int height;

    std::vector<RouteGraphVertex> vertices;
    std::vector<RouteGraphEdge> edges;

    // std::vector<std::shared_ptr<database::Pin> > vertices; // 16 * 10^8 ~= 1.6G
    // std::vector<XY<INDEX_T>> vertexPos;// 8 * 10^8 ~= 0.8G
    // std::vector<XY<INDEX_T>> vertexPosLow;// 8 * 10^8 ~= 0.8G
    // std::vector<XY<INDEX_T>> vertexPosHigh;
    // std::vector<COST_T> vertexCost;// 8 * 10^8 ~= 0.8G
    // std::vector<int> vertexCap;
    // std::vector<int> vertexMaxCap;
    // std::vector<std::vector<EdgeNode> > edges; // 24 * 10^8 + 8 * 4 * 1
    // std::vector<std::vector<int>> vertexId; // 24 * 10^5 + 4 * 10^8  ~=0 .4G
    // std::vector<std::vector<int>> clkVertexId; // 24 * 10^5 + 4 * 10^8  ~=0 .4G
    // std::vector<std::vector<int>> gswVertexId;  // 2G
    // std::vector<VertexType> vertexTypes;
    

    // int gridVertexNum;
};

} // namesapce router

#endif // ROUTEGRAPH_H
