#ifndef TIMING_GRAPH_H_
#define TIMING_GRAPH_H_

#include "utils/utils.h"

#include <vector>

namespace router {

enum TimingEdgeType {
    NETEDGE,
    INSTEDGE,
};

struct TimingEdge {
    INDEX_T source, sink;
    TimingEdgeType edgeType;
    COST_T edgeDelay;
    INDEX_T sourcePrev, sinkPrev;

    TimingEdge() {}
    TimingEdge(INDEX_T _source, INDEX_T _sink, TimingEdgeType type, COST_T delay) : source(_source), sink(_sink), edgeType(type), edgeDelay(delay) {}
};

struct TimingNode {
    INDEX_T vertexIdx;
    INDEX_T headSource;
    INDEX_T headSink;
    INDEX_T inputDegree, outputDegree;

    TimingNode() {}
    TimingNode(INDEX_T idx) : vertexIdx(idx), headSource(-1), headSink(-1), inputDegree(0), outputDegree(0) {}
};

class TimingGraph {
public:
    TimingGraph() {}
    ~TimingGraph() {}

    INDEX_T addVertex(int vertexIdx);
    void addEdge(int from, int to, TimingEdgeType edgeType, COST_T delay);
    TimingNode& getVertex(int idx) { return vertices[idx]; }
    TimingEdge& getEdge(int idx) { return edges[idx]; }
    COST_T getEdgeDelay(TimingEdge& edge);
    COST_T getEdgeDelay(int idx) { return getEdgeDelay(edges[idx]); }

    int getVertexNum() { return vertices.size(); }

    std::vector<TimingEdge>& getEdges() { return edges; }

private:
    std::vector<TimingEdge> edges;
    std::vector<TimingNode> vertices;

};

}

#endif //TIMING_GRAPH_H_