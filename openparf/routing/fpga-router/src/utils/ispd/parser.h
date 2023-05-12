#ifndef ISPD_PARSER_H
#define ISPD_PARSER_H
#include <vector>

#include "router/net.h"
#include "router/routegraph.h"
#include "database/builder_template.h"
#include "ispdnode.h"
using namespace router;
NodeType getNodeType(std::string type);
int getPinIdxInGraph(std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib, std::shared_ptr<RouteGraph> graph, ISPDNode node, std::string nodePin);
std::vector<std::shared_ptr<Net>> buildNetlist(const char* placefile,
                                               const char* netfile, 
                                               const char* nodefile, 
                                               std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib, 
                                               std::vector<std::vector<database::GridContent> >& layout,
                                               std::shared_ptr<RouteGraph> graph);
#endif //ISPD_PARSER_H