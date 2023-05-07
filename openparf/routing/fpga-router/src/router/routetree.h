#ifndef ROUTETREE_H
#define ROUTETREE_H
#include "database/pin.h"
#include "net.h"
#include "utils/utils.h"
#include "routegraph.h"

#include <unordered_map>
#include <unordered_set>
namespace router {
    class TreeNode {
    public:
        TreeNode() {}
        TreeNode(std::shared_ptr<Net> _net, int _nodeId)
            : net(_net), nodeId(_nodeId), father(nullptr), firstChild(nullptr), left(nullptr), right(nullptr), nodeDelay(0) {
                source = net->getSource();
            }
        TreeNode(std::shared_ptr<Net> _net, int _nodeId, COST_T delay)
            : net(_net), nodeId(_nodeId), father(nullptr), firstChild(nullptr), left(nullptr), right(nullptr), nodeDelay(delay) {
                source = net->getSource();
            }
         
        std::shared_ptr<Net> net;
        std::shared_ptr<TreeNode> father;
        std::shared_ptr<TreeNode> firstChild;
        std::shared_ptr<TreeNode> left;
        std::shared_ptr<TreeNode> right;
        int nodeId;
        int source;
        COST_T nodeDelay;
    };

    class RouteTree {
    public:

        RouteTree() {}

        void init(std::shared_ptr<RouteGraph> graph, std::vector<std::shared_ptr<Net>>& netlist);
        std::shared_ptr<TreeNode> addNode(std::shared_ptr<TreeNode> father, int nodeId, std::shared_ptr<Net> net);
        std::shared_ptr<TreeNode> addNode(std::shared_ptr<TreeNode> father, int nodeId, std::shared_ptr<Net> net, COST_T delay);
        void ripup(std::vector<std::shared_ptr<Net>>& netlist, bool expanding);

        std::shared_ptr<TreeNode> getNetRoot(std::shared_ptr<Net> net) { return netRoot[net]; }
        std::shared_ptr<Net> getNodeNet(int nodeId) { 
            if (treenodes[nodeId] == nullptr) return nullptr;
            return treenodes[nodeId]->net; 
        }
        std::shared_ptr<TreeNode> getTreeNodeByIdx(int nodeId) { return treenodes[nodeId]; }
        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<TreeNode>>& getNetRoots() { return netRoot; }
        COST_T getTotalWL();

        void eraseNode(std::shared_ptr<TreeNode> node);
        std::shared_ptr<RouteGraph> routeGraph() { return _graph; }


        friend class Pathfinder;
        friend class Router;
        friend class TimingGraph;
    private:
        COST_T congestAddCost;
        COST_T usedAddCost;

        std::shared_ptr<RouteGraph> _graph;
        bool ripupDfsSearch(std::shared_ptr<TreeNode> node, bool isDeleting, bool expanding);

        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<TreeNode>> netRoot;
        std::vector<std::shared_ptr<TreeNode>> treenodes;
        // std::vector<bool> congested;
    };
} // namespace router

#endif // ROUTETREE_H