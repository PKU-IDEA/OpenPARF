#ifndef VPRPARSER_H
#define VPRPARSER_H

#include "router/routegraph.h"
#include "router/net.h"
#include "router/pathfinder.h"

#include <vector>

namespace router {
    VertexType getVertexType(std::string typeName);
    void printVPRWirelength(std::vector<std::shared_ptr<Net>>& netlist, std::shared_ptr<RouteGraph> graph);
    std::shared_ptr<RouteGraph> parseRRGraph(const char* fileName);
    std::vector<std::shared_ptr<Net>> parseRouteFile(const char* fileName, std::shared_ptr<RouteGraph> graph);
} // namespace router

#endif //VPRPARSER_H