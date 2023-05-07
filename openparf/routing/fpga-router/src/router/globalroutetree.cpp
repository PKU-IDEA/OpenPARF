#include "globalroutetree.h"

#include <queue>

namespace router {

    void GlobalRouteTree::init(std::shared_ptr<GlobalRouteGraph> _globalGraph, std::shared_ptr<RouteGraph> _graph, std::vector<std::shared_ptr<Net>>& netlist) {
        globalGraph = _globalGraph;
        graph = _graph;
        // treenodes.resize(globalGraph->getVertexNum(), nullptr);
        for (auto net : netlist) {
            auto sourcePos = graph->getPos(net->getSource());
            int sourceIdx = globalGraph->getVertexIdx(sourcePos);
            netRoot[net] = std::make_shared<GlobalTreeNode>(net, sourceIdx);
        }
    }

    std::shared_ptr<GlobalTreeNode> GlobalRouteTree::addNode(std::shared_ptr<GlobalTreeNode> father, int edgeId, std::shared_ptr<Net> net) {
        int nodeId = globalGraph->getEdge(father->nodeId, edgeId).to;
        std::shared_ptr<GlobalTreeNode> node = std::make_shared<GlobalTreeNode>(net, nodeId);
        node->father = father;
        node->right = father->firstChild;
        if (father->firstChild != nullptr) father->firstChild->left = node;
        father->firstChild = node;
        globalGraph->decreaseCap(father->nodeId, edgeId);
        node->fatherEdgeId = edgeId;
        return node;
    }

    void GlobalRouteTree::eraseNode(std::shared_ptr<GlobalTreeNode> node) {
        if (node->father->firstChild == node) 
            node->father->firstChild = node->right;
        if (node->left != nullptr)
            node->left->right = node->right;
        if (node->right != nullptr)
            node->right->left = node->left;
        globalGraph->increaseCap(node->father->nodeId, node->fatherEdgeId);
        node->father = nullptr;
    }

    void GlobalRouteTree::ripup() {
        finished = true;
        for (auto it : netRoot) {
            ripupDfsSearch(it.second, false);
        }
        int vertexNum = globalGraph->getVertexNum();
        for (int i = 0; i < vertexNum; i++) {
            int degree = globalGraph->getDegree(i);
            for (int j = 0; j < degree; j++)
                globalGraph->setCongest(i, j, false);
        }
    }

    void GlobalRouteTree::ripupDfsSearch(std::shared_ptr<GlobalTreeNode> node, bool isDeleting) {
        if (node->father != nullptr) {
            if (globalGraph->getEdge(node->father->nodeId, node->fatherEdgeId).cap < -2 ||
                globalGraph->getEdge(node->father->nodeId, node->fatherEdgeId).congest) {
                    isDeleting = true;
                    globalGraph->setCongest(node->father->nodeId, node->fatherEdgeId, true);
                    finished = false;
                }
        }
        for (auto child = node->firstChild; child != nullptr; child = child->right)
            ripupDfsSearch(child, isDeleting);
        if (isDeleting) 
            eraseNode(node);
    }

    void GlobalRouteTree::initNetGlobalResult() {
        int totalWL = 0;
        FILE *f = fopen("gr.res", "w");

        for (auto it : netRoot) {
            auto net = it.first;
            auto root = it.second;
            std::queue<std::shared_ptr<GlobalTreeNode>> q;
            q.push(root);
            int wl = 0;
            while (!q.empty()) {
                auto node = q.front();
                q.pop();
                if (node->father != nullptr) {
                    int prevId = node->father->nodeId;
                    int currId = node->nodeId;
                    int prevX = prevId / graph->getHeight(), prevY = prevId % graph->getHeight();
                    int currX = currId / graph->getHeight(), currY = currId % graph->getHeight();
                    int startY = std::min(currY, prevY);
                    int endY   = std::max(currY, prevY);
                    int startX = std::min(currX, prevX);
                    int endX   = std::max(currX, prevX);
                    totalWL += endY - startY + endX - startX;
                    wl += endY - startY + endX - startX;
                    if (net->getName() == "sig_292") {
                        fprintf(f, "Net %s route point (%d, %d)->(%d, %d)\n", net->getName().c_str(), prevX, prevY, currX, currY);
                    }
                    for (int i = startX; i <= endX; i++)
                        for (int j = startY; j <= endY; j++)
                            net->addGlobalRouteResult(i, j);
                }
                for (auto child = node->firstChild; child != nullptr; child = child->right) {
                    q.push(child);
                }
            }
            int source = net->getSource();
            net->addGlobalRouteResult(graph->getPos(source).X(), graph->getPos(source).Y());
            net->useGlobalResult(true);
            fprintf(f, "Net %s, GRWL: %d\n", net->getName().c_str(), wl);
        }

        fclose(f);
        std::cout << "total GR WL : " << totalWL << std::endl;
    }

} // namespace router