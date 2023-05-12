#ifndef TILE_ROUTER_H
#define TILE_ROUTER_H

#include "routegraph.h"
// #include "router.h"
#include "database/builder_template.h"

#include <gurobi_c++.h>

namespace router {

class Router;

class TileRouter {
public: 
    TileRouter() {}
    TileRouter(std::shared_ptr<RouteGraph> _graph, std::shared_ptr<database::GridLayout> _layout) 
     : graph(_graph), layout(_layout){
        gridNetlist.resize(graph->getWidth());
        for (int i = 0 ; i < gridNetlist.size(); i++) {
            gridNetlist[i].resize(graph->getHeight());
        }
    }

    void tileRoute();
    void buildLocalGraph(std::string moduleName);
    void setRouter(Router* _router) { router = _router; }
    
private:

    void addLocalGraphVertex(std::shared_ptr<database::Module> module);
    void addLocalGraphEdge(std::shared_ptr<database::Module> module);

    void ILPRoute(GRBEnv& env, int x, int y);
    
    Router* router;
    std::shared_ptr<RouteGraph> graph;
    std::shared_ptr<database::GridLayout> layout;
    std::vector<int> localgraphIdx;

    std::vector<std::vector<std::vector<std::shared_ptr<Net>>>> gridNetlist;
    std::shared_ptr<LocalRouteGraph> localgraph;
};

}

#endif //TILE_ROUTER_H