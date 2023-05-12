#ifndef XArchPARSER_H
#define XArchPARSER_H
#include <vector>

#include "router/net.h"
#include "router/routegraph.h"
#include "database/builder_template.h"
#include "xarchnode.h"
using namespace router;
class XArchParser {
public:
    XArchNodeType getNodeType(std::string type);
    int getPinIdxInGraph(std::vector<std::vector<database::GridContent> >& layout, std::shared_ptr<RouteGraph> graph, XArchNode node, std::string nodePin);
    std::vector<std::shared_ptr<Net>> buildNetlist(const char* placefile,
                                                const char* netfile,
                                                const char* nodefile,
                                                std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib,
                                                std::vector<std::vector<database::GridContent> >& layout,
                                                std::shared_ptr<RouteGraph> graph);
    void buildInst(std::vector<std::vector<database::GridContent>>& layout, std::shared_ptr<RouteGraph> graph, XArchNode node, std::string nodeName);
};
#endif //XArchPARSER_H