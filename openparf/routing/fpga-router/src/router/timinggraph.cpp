#include "timinggraph.h"
#include "pathfinder.h"

namespace router {
INDEX_T TimingGraph::addVertex(int vertexIdx) {
    vertices.emplace_back(vertexIdx);
    return vertices.size() - 1;
}

void TimingGraph::addEdge(int from, int to, TimingEdgeType edgeType, COST_T delay) {
    assert(from < vertices.size());
    assert(to < vertices.size());
    edges.emplace_back(from, to, edgeType, delay);
    int edgeIdx = edges.size() - 1;
    edges[edgeIdx].sourcePrev = vertices[from].headSource;
    vertices[from].headSource = edgeIdx;
    edges[edgeIdx].sinkPrev = vertices[to].headSink;
    vertices[to].headSink = edgeIdx;
    vertices[from].outputDegree++;
    vertices[to].inputDegree++;
}

COST_T TimingGraph::getEdgeDelay(TimingEdge& edge) {
    auto& routetree = Pathfinder::routetree;
    auto node = routetree.getTreeNodeByIdx(vertices[edge.sink].vertexIdx);
    
    // if (node->nodeId == 20514401) {
    //     std::cout << "getEdgeDelay: " << ' ' << node->nodeId << ' ' << node->nodeDelay << std::endl;
    // }
    switch (edge.edgeType)
    {
    case INSTEDGE:
        return edge.edgeDelay;
        break;
    case NETEDGE:
        if (node == nullptr) {
            std::cout << "PIN " << routetree._graph->getVertexByIdx(vertices[edge.sink].vertexIdx) ->getName() << ' ' << routetree._graph->getPos(vertices[edge.sink].vertexIdx).X() << ' ' << routetree._graph->getPos(vertices[edge.sink].vertexIdx).Y() << " unrouted!" << std::endl;
        }
        assert(node != nullptr);
        return node->nodeDelay;
        break;
    default:
        break;
    }
}
}