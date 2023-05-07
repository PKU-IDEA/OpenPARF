#ifndef ROUTEGRAPHBUILDER_H
#define ROUTEGRAPHBUILDER_H

#include "routegraph.h"
// #include "net.h"
#include "database/module.h"
#include "utils/utils.h"
#include "database/builder_template.h"

#include <memory>

namespace router {

class RouteGraphBuilder {
public:
    RouteGraphBuilder() {}
    RouteGraphBuilder(std::shared_ptr<database::GridLayout> layout) : gridLayout(layout) {}

    virtual std::shared_ptr<RouteGraph> run();
    void buildGraphRecursive(std::shared_ptr<database::Module> currentModule, std::shared_ptr<RouteGraph> graph, int x, int y, int width, int height);
    void addConnectRecursive(std::shared_ptr<database::Module> currentModule, std::shared_ptr<RouteGraph> graph, int x, int y);

    static int globalGraphMaxWidth;
protected:
    std::shared_ptr<database::GridLayout> gridLayout;
    std::vector<std::unordered_map<std::string, int>> gswVertexId;
};


} // namespace router

#endif // ROUTEGRAPHBUILDER_H