#ifndef GLOBALROUTETREE_H
#define GLOBALROUTETREE_H
#include "database/pin.h"
#include "net.h"
#include "utils/utils.h"
#include "routegraph.h"
#include "globalroutegraph.h"

#include <unordered_map>
#include <unordered_set>
namespace router {
    class GlobalTreeNode {
    public:
        GlobalTreeNode() {}
        GlobalTreeNode(std::shared_ptr<Net> _net, int _nodeId)
            : net(_net), nodeId(_nodeId), father(nullptr), firstChild(nullptr), left(nullptr), right(nullptr) {}
        
        std::shared_ptr<Net> net;
        std::shared_ptr<GlobalTreeNode> father;
        std::shared_ptr<GlobalTreeNode> firstChild;
        std::shared_ptr<GlobalTreeNode> left;
        std::shared_ptr<GlobalTreeNode> right;
        int nodeId;
        int fatherEdgeId;
    };

    class GlobalRouteTree {
    public:

        GlobalRouteTree() {}

        void init(std::shared_ptr<GlobalRouteGraph> globalGraph, std::shared_ptr<RouteGraph> graph, std::vector<std::shared_ptr<Net>>& netlist);
        std::shared_ptr<GlobalTreeNode> addNode(std::shared_ptr<GlobalTreeNode> father, int edgeId, std::shared_ptr<Net> net);
        void ripup();

        std::shared_ptr<GlobalTreeNode> getNetRoot(std::shared_ptr<Net> net) { return netRoot[net]; }
        // std::shared_ptr<Net> getNodeNet(int nodeId) { 
        //     if (treenodes[nodeId] == nullptr) return nullptr;
        //     return treenodes[nodeId]->net; 
        // }
        // std::shared_ptr<GlobalTreeNode> getTreeNodeByIdx(int nodeId) { return treenodes[nodeId]; }
        bool finish() { return finished; }

        void initNetGlobalResult();

        friend class Pathfinder;
        friend class Router;
    private:
        COST_T congestAddCost;
        COST_T usedAddCost;

        bool finished; 

        std::shared_ptr<RouteGraph> graph;
        std::shared_ptr<GlobalRouteGraph> globalGraph;

        void eraseNode(std::shared_ptr<GlobalTreeNode> node);
        void ripupDfsSearch(std::shared_ptr<GlobalTreeNode> node, bool isDeleting);

        // std::unordered_map<std::shared_ptr<Net>, bool> congestNets;
        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<GlobalTreeNode>> netRoot;
        // std::vector<std::shared_ptr<GlobalTreeNode>> treenodes;
    };
} // namespace router

#endif // GLOBALROUTETREE_H