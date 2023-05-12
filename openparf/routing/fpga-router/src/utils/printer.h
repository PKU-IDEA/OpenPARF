#ifndef PRINTER_H
#define PRINTER_H
#include <memory>
#include <vector>

#include <pugixml/pugixml.hpp>
#include "router/net.h"
#include "router/routegraph.h"
namespace router {

void printRouteResult(std::vector<std::shared_ptr<router::Net>> &netlist,
        std::string                                              fileName,
        std::shared_ptr<router::RouteGraph>                      graph);

void getPinGrid(const std::string &pinName, int &x, int &y);

}   // namespace router
#endif   // PRINTER_H