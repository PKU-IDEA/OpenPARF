#ifndef ROUTER_H
#define ROUTER_H
#include <vector>
#include <memory>

#include "pathfinder.h"
#include "net.h"
#include "database/module.h"
#include "database/builder_template.h"
#include <pugixml/pugixml.hpp>
#include "utils/utils.h"


namespace router {

class Router {
public:
    Router() {}
    Router(std::shared_ptr<RouteGraph> _graph, std::shared_ptr<database::GridLayout> layout, int mttype) : graph(_graph), layout_(layout), MTType(mttype) {}
    Router(std::shared_ptr<RouteGraph> _graph, std::shared_ptr<database::GridLayout> layout, std::string inNetFile, int mttype);
    // Router(std::shared_ptr<database::Module> topModule);
    // Router(std::string netlistFile, std::shared_ptr<database::Module> topModule);

    void addNet(std::shared_ptr<Net> net) {netlist.push_back(net);}
    void run();
    // void ripup(std::shared_ptr<Net> net);
    std::vector<std::shared_ptr<Net> >& getNetlist() { return netlist; }
    void buildPinMap(std::unordered_map<std::string, std::shared_ptr<database::Pin>>& pinMap, std::shared_ptr<database::Module> currentModule);

    void routeTileNets();


    std::string grFileName;
    static int maxRipupIter;
    static int printCongestMapIter;
private:
    std::vector<std::shared_ptr<Net> > netlist;
    std::shared_ptr<RouteGraph> graph;
    std::shared_ptr<database::GridLayout> layout_;

    static const int SingleThread = 0;
    static const int StaticSchedule = 1;
    static const int DynamicSchedule = 2;

    int MTType;

};


// std::shared_ptr<database::Pin> findPin(std::string pinName, std::shared_ptr<database::Module> topModule);

} // namespace router


#endif // ROUTER_H