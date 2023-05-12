#ifndef NETGENERATOR_H
#define NETGENERATOR_H
#include "router/net.h"
#include "router/routegraph.h"

#include <memory>
#include <vector>

std::vector<std::shared_ptr<router::Net>> generateNetlistRandomly(std::shared_ptr<router::RouteGraph> graph);
std::shared_ptr<router::Net> generateNetRandomly(std::shared_ptr<router::RouteGraph> topModule); 
#endif //NETGENERATOR_H