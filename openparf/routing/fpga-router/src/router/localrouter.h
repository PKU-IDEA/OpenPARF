#ifndef LOCALROUTER_H
#define LOCALROUTER_H

#include "routegraph.h"
#include "routetree.h"
#include "net.h"
#include "database/builder_template.h"
#include <pugixml/pugixml.hpp>
#include "mcf_solver/mcf.h"

#include <lemon/network_simplex.h>
#include <lemon/concepts/digraph.h>
#include <lemon/list_graph.h>
#include <lemon/lgf_writer.h>
#include <gurobi_c++.h>



namespace router {

class GridNet : public Net {
public:
    GridNet();
    GridNet(std::string _name, std::shared_ptr<Net> net) : Net(_name), originNet(net) {}

    void setSource(int _source, std::shared_ptr<TreeNode> sourceNode) {
        Net::setSource(_source);
        sourceTreeNode = sourceNode;
    }

    void addSink(int _sink, std::shared_ptr<TreeNode> sinkNode) {
        Net::addSink(_sink);
        sinkTreeNodes.push_back(sinkNode);
        sinkTreeNodeSet.insert(sinkNode);
    }

    std::shared_ptr<TreeNode> getSourceTreeNode() { return sourceTreeNode; }
    std::shared_ptr<TreeNode> getSinkTreeNodeByIdx(int idx) { return sinkTreeNodes[idx]; }
    bool isSinkTreeNode(std::shared_ptr<TreeNode> node) { return (sinkTreeNodeSet.find(node) != sinkTreeNodeSet.end()); }
    std::shared_ptr<Net> getOriginNet() { return originNet; }
private:

    std::shared_ptr<TreeNode> sourceTreeNode;
    std::vector<std::shared_ptr<TreeNode> > sinkTreeNodes;
    std::set<std::shared_ptr<TreeNode> > sinkTreeNodeSet;
    std::shared_ptr<Net> originNet;
};

class LocalRouter {
public:
    LocalRouter() {}
    LocalRouter(std::shared_ptr<RouteGraph> _graph) : graph(_graph), env(){
        gridNetlist.resize(graph->getWidth());
        for (int i = 0 ; i < gridNetlist.size(); i++) {
            gridNetlist[i].resize(graph->getHeight());
        }
         env.set(GRB_IntParam_OutputFlag, 0);
         env.set(GRB_IntParam_Threads, 1);
         env.set(GRB_IntParam_MIPFocus, 1);
    }
    // void loadRouteResultFromXML(std::string xmlFile);
    void loadRouteResultFromRouteTree(RouteTree& routeTree);

    void testRun();
    std::shared_ptr<LocalRouteGraph> dumpRouteGraph(int x, int y);
    void dumpLocalGraphAndPrint(int x, int y, std::string fileName);
    void tryILP(int x, int y);

    void dumpLemonDiGraph(std::shared_ptr<LocalRouteGraph> localgraph, int x, int y);
    void tryMCMCFRoute(int x, int y);
    void setLayout(std::shared_ptr<database::GridLayout> layout_) { layout = layout_; }
private:
    void routeTreeDfs(std::shared_ptr<TreeNode> node, std::shared_ptr<GridNet> net);
    void ripupDfs(std::shared_ptr<TreeNode> node, std::shared_ptr<GridNet> net);
    void buildGridNetDfs(std::shared_ptr<TreeNode> node, int x, int y, std::shared_ptr<GridNet> net, std::vector<std::shared_ptr<GridNet> >& netlist, std::unordered_map<int, int>& visited_vertex);

    std::shared_ptr<RouteGraph> graph;
    std::vector<std::vector<std::vector<std::shared_ptr<GridNet>>>> gridNetlist;
    // std::vector<std::shared_ptr<Net> > netlist;
    std::set<std::pair<int, int> > congestPos;
    std::shared_ptr<database::GridLayout> layout;
    GRBEnv env;

};

}

#endif