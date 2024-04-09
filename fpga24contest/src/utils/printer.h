#ifndef PRINTER_H
#define PRINTER_H
#include"thirdparty/pugixml/pugixml.hpp"
#include"router/net.h"
#include "router/routegraph.h"
#include <memory>
#include <vector>
namespace router {

void printRouteResult(std::vector<std::shared_ptr<router::Net>> &netlist, std::string fileName,
                      std::shared_ptr<router::RouteGraph> graph);

void printNetRouteResult(std::shared_ptr<router::Net> &net, std::string fileName,
                      std::shared_ptr<router::RouteGraph> graph, int iter);

void getPinGrid(const std::string &pinName, int &x, int &y);

} // namespace router
#endif //PRINTER_H