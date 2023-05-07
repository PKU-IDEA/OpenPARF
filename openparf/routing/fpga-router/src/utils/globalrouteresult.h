#ifndef GLOBAL_ROUTE_RESULT_H
#define GLOBAL_ROUTE_RESULT_H

#include "router/net.h"

namespace router {
    void printGlobalRouteResult(std::vector<std::shared_ptr<Net>>& netlist, std::string fileName);
    void loadGlobalRouteResult(std::vector<std::shared_ptr<Net>>& netlist, std::string fileName);
} // namespace router


#endif