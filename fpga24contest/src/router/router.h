#ifndef ROUTER_H
#define ROUTER_H
#include <vector>
#include <memory>

#include "pathfinder.h"
#include "net.h"
#include "../thirdparty/pugixml/pugixml.hpp"
#include "../utils/utils.h"


namespace router {

class Router {
public:
    Router() {}
    Router(std::shared_ptr<RouteGraph> _graph) : graph(_graph) {}
    // Router(std::shared_ptr<database::Module> topModule);
    // Router(std::string netlistFile, std::shared_ptr<database::Module> topModule); 

    void addNet(std::shared_ptr<Net> net) {netlist.push_back(net);}
    // void ripup(std::shared_ptr<Net> net);
    std::vector<std::shared_ptr<Net> >& getNetlist() { return netlist; }

    void routeSingleNet(std::shared_ptr<Net> net);
    void runTaskflow();

private:
    std::vector<std::shared_ptr<Net> > netlist;
    std::shared_ptr<RouteGraph> graph;

    static const int SingleThread = 0;
    static const int StaticSchedule = 1;
    static const int DynamicSchedule = 2;
    int MTType;

}; 


// std::shared_ptr<database::Pin> findPin(std::string pinName, std::shared_ptr<database::Module> topModule);

} // namespace router


#endif // ROUTER_H