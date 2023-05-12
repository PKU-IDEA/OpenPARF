#include "globalroutegraph.h"

#include <cmath>
#include <iostream>

namespace router {
    GlobalRouteGraph::GlobalRouteGraph(int _width, int _height) {
        width = _width;
        height = _height;
        vertexNum = width * height;

        edges.resize(width * height);
    }

    void GlobalRouteGraph::addEdge(int startX, int startY, int endX, int endY) {
        int startId = startX * height + startY;
        int endId = endX * height + endY;
        // std::cout << startId << ' ' << endId << std::endl;
        for (int i = 0; i < edges[startId].size(); i++) {
            if (edges[startId][i].to == endId) {
                edges[startId][i].cap++;
                return;
            }
        }
        edges[startId].push_back(GlobalRouteGraphEdgeNode(endId, 1, abs(endX - startX) + abs(endY - startY)));
    }

    int GlobalRouteGraph::getVertexIdx(XY<INDEX_T> pos) {
        return pos.X() * height + pos.Y();
    }
}