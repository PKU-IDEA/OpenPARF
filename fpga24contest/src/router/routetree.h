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
            : net(_net), nodeId(_nodeId), father(nullptr), firstChild(nullptr), left(nullptr), right(nullptr) {
                source = net->getSource();
            }
         
        std::shared_ptr<Net> net;
        std::shared_ptr<TreeNode> father;
        std::shared_ptr<TreeNode> firstChild;
        std::shared_ptr<TreeNode> left;
        std::shared_ptr<TreeNode> right;
        int nodeId;
        int source;
    };

    class RouteTree {
    public:

        RouteTree() {}

        void init(std::shared_ptr<RouteGraph> graph, std::vector<std::shared_ptr<Net>>& netlist);
        std::shared_ptr<TreeNode> addNode(std::shared_ptr<TreeNode> father, int nodeId, std::shared_ptr<Net> net);
        std::shared_ptr<TreeNode> addNodeReverse(std::shared_ptr<TreeNode>& father, std::shared_ptr<TreeNode> child, int nodeId, std::shared_ptr<Net> net, bool isVirtual);

        void ripup(std::vector<std::shared_ptr<Net>>& netlist, bool expanding);
        std::shared_ptr<TreeNode> getNetRoot(std::shared_ptr<Net> net) { return netRoot[net]; }
        std::shared_ptr<Net> getNodeNet(int nodeId) { 
            if (treenodes[nodeId] == nullptr) return nullptr;
            return treenodes[nodeId]->net; 
        }
        std::shared_ptr<TreeNode> getTreeNodeByIdx(int nodeId) { return treenodes[nodeId]; }
        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<TreeNode>>& getNetRoots() { return netRoot; }
        COST_T getTotalWL();
        COST_T dumpStatisticData(std::shared_ptr<Net> net);
        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<TreeNode>> const& getNetRoot() const { return netRoot; }

        void eraseNode(std::shared_ptr<TreeNode> node);


        friend class Pathfinder;
        friend class Router;
    private:
        COST_T congestAddCost;
        COST_T usedAddCost;

        std::shared_ptr<RouteGraph> _graph;
        bool ripupDfsSearch(std::shared_ptr<TreeNode> node, bool isDeleting, bool expanding, std::shared_ptr<Net> net);
        std::unordered_map<std::shared_ptr<Net>, std::shared_ptr<TreeNode>> netRoot;
        std::vector<std::shared_ptr<TreeNode>> treenodes;
        // std::vector<bool> congested;
    };
} // namespace router

#endif // ROUTETREE_H