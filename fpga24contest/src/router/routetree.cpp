#include "routetree.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <queue>
namespace router {
    void RouteTree::init(std::shared_ptr<RouteGraph> graph, std::vector<std::shared_ptr<Net>>& netlist) {
        int vertexNum = graph->getVertexNum();
        // treenodes.clear();
        treenodes.resize(vertexNum, nullptr);

        // netRoot.clear();

        for (auto net : netlist) {
            netRoot[net] = std::make_shared<TreeNode> (net, net->getSource());
            treenodes[net->getSource()] = netRoot[net];
        }
        // congested.resize(vertexNum, false);

        _graph = graph;
        usedAddCost = 0;
        congestAddCost = 2;
    }

    std::shared_ptr<TreeNode> RouteTree::addNode(std::shared_ptr<TreeNode> father, int nodeId, std::shared_ptr<Net> net) {
        if (treenodes[nodeId] != nullptr && treenodes[nodeId]->net == net) {
            std::cout << "[Warning] Attempting to add an existing node " << nodeId << " into net " << net->getName() << "'s root tree" << std::endl;
            std::cout << "          Pin " << nodeId << " (" << _graph->getPos(nodeId).X() << ", " << _graph->getPos(nodeId).Y() << ") name " << _graph->getVertexByIdx(nodeId)->getName() << std::endl;
            return treenodes[nodeId];
        }
        std::shared_ptr<TreeNode> node = std::shared_ptr<TreeNode>(new TreeNode(net, nodeId));
        node->father = father;
        node->right = father->firstChild;
        if (father->firstChild != nullptr) father->firstChild->left = node;
        father->firstChild = node;
        _graph->addVertexCap(nodeId, -1);

        treenodes[nodeId] = node;
        return node;
    }

 
    std::shared_ptr<TreeNode> RouteTree::addNodeReverse(std::shared_ptr<TreeNode> &father, std::shared_ptr<TreeNode> child, int nodeId, std::shared_ptr<Net> net, bool isVirtual) {
        if (treenodes[nodeId] != nullptr && treenodes[nodeId]->net == net) {
            std::cout << "[Warning] Attempting to add an existing node " << nodeId << " into net " << net->getName() << "'s root tree" << std::endl;
            std::cout << "pin name " << _graph->getVertexByIdx(nodeId)->getName() << std::endl;
            return treenodes[nodeId];
        }

        if (isVirtual) {
            child->father = father;
            child->right = father->firstChild;
            if (father->firstChild != nullptr) father->firstChild->left = child;
            father->firstChild = child;
            _graph->addVertexCap(nodeId, -1);

            child->net = net;

            treenodes[nodeId] = child;
            return child;
        } else {
            std::shared_ptr<TreeNode> node = std::shared_ptr<TreeNode>(new TreeNode(net, nodeId));
            node->father = father;
            node->right = father->firstChild;
            if (father->firstChild != nullptr) father->firstChild->left = node;
            father->firstChild = node;
            _graph->addVertexCap(nodeId, -1);            
            treenodes[nodeId] = node;
            return node;
        }

    }

    void RouteTree::eraseNode(std::shared_ptr<TreeNode> node) {
        if (node->father != nullptr && node->father->firstChild == node) 
            node->father->firstChild = node->right;
        if (node->left != nullptr)
            node->left->right = node->right;
        if (node->right != nullptr)
            node->right->left = node->left;
        node->father = nullptr;
        if (treenodes[node->nodeId] == node) treenodes[node->nodeId] = nullptr;
        _graph->addVertexCap(node->nodeId, 1);
    }

    void RouteTree::ripup(std::vector<std::shared_ptr<Net>>& netlist, bool expanding) {

        _graph->updateVertexCost();
        for (int i = netlist.size() - 1; i >= 0; i--) { // ripup large net first
            // std::cout << it.first->getName() << ' ' << it.second << std::endl;
                auto net = netlist[i];
                net->clearCongestedVertices();
                if (ripupDfsSearch(netRoot[net], false, expanding, net))
                  net->setRouteStatus(CONGESTED);
            }

        int vertexNum = _graph->getVertexNum();
        // for (int i = 0; i < vertexNum; i++)
        //     congested[i] = false;
        // std::ofstream of1(fileName1), of2(fileName2);
        // for (int i = 0; i < _graph->getWidth(); i++) {
        //     for (int j = 0; j < _graph->getHeight(); j++) {
        //         if (j != 0) of1 << ' ', of2 << ' ';
        //         of1 << o_fu_byp_useCnt[i][j];
        //         of2 << o_gsw_switch_useCnt[i][j];
        //     }
        //     of1 << std::endl;
        //     of2 << std::endl;
        // }
        // congestAddCost *= 0.5;
    }

    bool RouteTree::ripupDfsSearch(std::shared_ptr<TreeNode> node, bool isDeleting, bool expanding, std::shared_ptr<Net> net) {
        bool res = false;
        // if (_graph->getVertexByIdx(node->nodeId)->getName().find("o_fu_byp") != std::string::npos)
        //     o_fu_byp_useCnt[_graph->getPos(node->nodeId).X()][_graph->getPos(node->nodeId).Y()]++;
        // if (_graph->getVertexByIdx(node->nodeId)->getName().find("o_gsw_switch") != std::string::npos)
        //     o_gsw_switch_useCnt[_graph->getPos(node->nodeId).X()][_graph->getPos(node->nodeId).Y()]++;、
        // if (node->net->getName() == "net_118568" || node->net->getName() == "net_116700")
        // if (RouteGraph::debugging)
        //     std::cout << node->net->getName() << ' ' << _graph->getVertexByIdx(node->nodeId)->getName() << " isDeleting: " << isDeleting << std::endl;
        // if (node->nodeId == 27655277) {
        //     std::cout << "ripUpDfsSearch: " << node->nodeId << ' ' << isDeleting << ' ' << expanding << ' ' << net->getName() << std::endl;
        // }
        if (isDeleting || _graph->getVertexCap(node->nodeId) < 0) {
            isDeleting = res = true;
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                res |= ripupDfsSearch(child, true, expanding, net);
            }
            // if (_graph->getVertexCap(node->nodeId) <= 0)
            // _graph->addVertexCost(node->nodeId, -usedAddCost);
            eraseNode(node);
        }
        for (auto child = node->firstChild; child != nullptr; child = child->right)
            res |= ripupDfsSearch(child, isDeleting, expanding, net);
        // if (node->net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]" && node->nodeId == 35653)
        //     std::cout << node->firstChild << ' ' << isDeleting << std::endl;
        if (node->firstChild == nullptr && !node->net->isSink(node->nodeId) && !isDeleting && !(node->net->getSource() == node->nodeId))
            eraseNode(node);
        return res;
    }

    // used for uniform GSW position

    COST_T RouteTree::getTotalWL() {
        COST_T totalWL = 0;
        int totalNum  = 0;
        for (auto it : netRoot) {

            totalNum++;

            // if (it.first->getName() != "net_283") continue;
            // if (it.first->getName() != "net_553") continue;
            // if ((it.first->getName() != "net_283") && (it.first->getName() != "net_553")) continue;
            // std::cout << "back trace net " << it.first->getName() << std::endl;

            COST_T netWL = 0;
            std::queue<std::shared_ptr<TreeNode>> q;
            q.push(it.second);

            int iter = 0;
            while(!q.empty()) {
                auto now = q.front();
                q.pop();
                int childNum = 0;
                for (auto child = now->firstChild; child != nullptr; child = child->right) {
                    auto posS = _graph->getPos(now->nodeId);
                    auto posT = _graph->getPos(child->nodeId);
                    netWL += abs(posT.X() - posS.X()) + abs(posT.Y() - posS.Y());
                    q.push(child);
                    // std::cout << child->nodeId << " ";
                    childNum++;

                    if (childNum > 100) {
                        std::cout << "childNum overflow in net " << it.first->getName() << " sinkNum " << it.first->getSinkSize() << " totalNum " << totalNum << std::endl;
                        break;
                    }
                }
                // std::cout << std::endl;

                // std::cout << "iter " << iter << " child num " << childNum << " nodeId " << now->nodeId;
                // if (now->father != nullptr) {
                //     std::cout << " fatherId " << now->father->nodeId << std::endl;
                // } else {
                //     std::cout << std::endl;
                // }

                if (iter > 100000) {
                    std::cout << "net " << it.first->getName() << " failed" << std::endl;
                    exit(0);
                }

                iter++;
            }

            // if (it.first->getSinkSize() > 1000) {
            //     std::cout << "Large Sinks " << it.first->getSinkSize() << " NET " << it.first->getName() << " WL: " << netWL << " Ripup-Nodes: " << it.first->getErasedNodes() << "\n";
            // }

            it.first->setNetWL(netWL);

            totalWL += netWL;
        }
        return totalWL;
    }



    COST_T RouteTree::dumpStatisticData(std::shared_ptr<Net> net) {
        COST_T totalWL = 0;
        int totalSteinerPointNum  = 0;
        int totalSteinerPointGSWNum  = 0;
        int gsw1 = 0;
        int gsw2 = 0;
        int gsw6 = 0;
        COST_T totalSteinerPointGSWLength = 0;

        auto it = netRoot[net];
        std::queue<std::shared_ptr<TreeNode>> q;
        q.push(it);

        int iter = 0;
        while(!q.empty()) {
            auto now = q.front();
            q.pop();
            int childNum = 0;
            for (auto child = now->firstChild; child != nullptr; child = child->right) {
                q.push(child);
                childNum++;
            }

            iter++;
        }

        std::cout << "dump statistic data iter " << iter << std::endl;


        if (net->getSinkSize() > 1000) {
            std::cout << "Large Sinks " << net->getSinkSize() << " Net " << net->getName() << " totalSPNum: " << totalSteinerPointNum << " totalSPGSWNum " << totalSteinerPointGSWNum << " totalSPGSWLength " << totalSteinerPointGSWLength << " averageSPGSWLength " << (double)totalSteinerPointGSWLength/totalSteinerPointGSWNum << " GSW1 " << gsw1 << " GSW2 " << gsw2 << " GSW6 " << gsw6 << "\n";
        }

        return 0;

    }


}
