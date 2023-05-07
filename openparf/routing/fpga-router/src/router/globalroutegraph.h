#ifndef GLOBALROUTEGRAPH_H
#define GLOBALROUTEGRAPH_H

#include "utils/utils.h"

#include <vector>

namespace router {
    class GlobalRouteGraphEdgeNode {
    public:
        GlobalRouteGraphEdgeNode() {}
        GlobalRouteGraphEdgeNode(int t, int c, COST_T l) : to(t), cap(c), length(l), congest(false) {}

        int to;
        int cap;
        COST_T length;
        bool congest;
    };
    class GlobalRouteGraph {
    public:
        GlobalRouteGraph() {}
        GlobalRouteGraph(int _width, int _height);
        int getVertexIdx(XY<INDEX_T> pos);
        void addEdge(int startX, int startY, int endX, int endY);
        GlobalRouteGraphEdgeNode getEdge(int posX, int posY, int edgeIdx);
        GlobalRouteGraphEdgeNode getEdge(int vertexIdx, int edgeIdx) { return edges[vertexIdx][edgeIdx];}
        int getDegree(int vertexIdx) { return edges[vertexIdx].size();}
        int getVertexNum() { return vertexNum; }

        void increaseCap(int vertexIdx, int edgeIdx) {
            edges[vertexIdx][edgeIdx].cap++;
        }

        void decreaseCap(int vertexIdx, int edgeIdx) {
            edges[vertexIdx][edgeIdx].cap--;
            if (edges[vertexIdx][edgeIdx].cap <= 4)
                edges[vertexIdx][edgeIdx].length++;
        }

        void setCongest(int vertexIdx, int edgeIdx, bool cong) {
            edges[vertexIdx][edgeIdx].congest = cong;
        }
        bool getCongest(int vertexIdx, int edgeIdx) {
            return edges[vertexIdx][edgeIdx].congest;
        }
    private:
        int width;
        int height;
        int vertexNum;
        
        std::vector<std::vector<GlobalRouteGraphEdgeNode> > edges;
    };
}

#endif