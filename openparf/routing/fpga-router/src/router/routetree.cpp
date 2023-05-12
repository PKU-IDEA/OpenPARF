#include "routetree.h"
#include "pathfinder.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <queue>
namespace router {
    void RouteTree::init(std::shared_ptr<RouteGraph> graph, std::vector<std::shared_ptr<Net>>& netlist) {
        int vertexNum = graph->getVertexNum();
        treenodes.resize(vertexNum, nullptr);

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
            std::cout << "[Warning] Attempting to add an existing node " << nodeId << " into net " << net->getName() << "'s root tree";
            return treenodes[nodeId];
        }
        std::shared_ptr<TreeNode> node = std::shared_ptr<TreeNode>(new TreeNode(net, nodeId));
        node->father = father;
        node->right = father->firstChild;
        if (father->firstChild != nullptr) father->firstChild->left = node;
        father->firstChild = node;
        _graph->addVertexCap(nodeId, -1);
        // if (_graph->getVertexCap(nodeId) < 0) congested[nodeId] = true;
            // _graph->addVertexCost(nodeId, usedAddCost);
            // std::cout << _graph->getVertexByIdx(nodeId)->getName() << std::endl;
            // getchar();
        
        treenodes[nodeId] = node;
        return node;
    }

    std::shared_ptr<TreeNode> RouteTree::addNode(std::shared_ptr<TreeNode> father, int nodeId, std::shared_ptr<Net> net, COST_T delay) {
        if (treenodes[nodeId] != nullptr && treenodes[nodeId]->net == net) {
            std::cout << "[Warning] Attempting to add an existing node " << nodeId << " into net " << net->getName() << "'s root tree";
            return treenodes[nodeId];
        }
        std::shared_ptr<TreeNode> node = std::shared_ptr<TreeNode>(new TreeNode(net, nodeId, delay));
        node->father = father;
        node->right = father->firstChild;
        if (father->firstChild != nullptr) father->firstChild->left = node;
        father->firstChild = node;
        _graph->addVertexCap(nodeId, -1);
        // if (_graph->getVertexCap(nodeId) < 0) congested[nodeId] = true;
            // _graph->addVertexCost(nodeId, usedAddCost);
            // std::cout << _graph->getVertexByIdx(nodeId)->getName() << std::endl;
            // getchar();
        
        treenodes[nodeId] = node;
        return node;
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

        int congestNum = _graph->updateVertexCost();
        if (!congestNum) return;
        for (int i = netlist.size() - 1; i >= 0; i--) {
            // std::cout << it.first->getName() << ' ' << it.second << std::endl;
                auto net = netlist[i];
                if (ripupDfsSearch(netRoot[net], false,expanding))
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

    bool RouteTree::ripupDfsSearch(std::shared_ptr<TreeNode> node, bool isDeleting, bool expanding) {
        bool res = false;
        // if (_graph->getVertexByIdx(node->nodeId)->getName().find("o_fu_byp") != std::string::npos)
        //     o_fu_byp_useCnt[_graph->getPos(node->nodeId).X()][_graph->getPos(node->nodeId).Y()]++;
        // if (_graph->getVertexByIdx(node->nodeId)->getName().find("o_gsw_switch") != std::string::npos)
        //     o_gsw_switch_useCnt[_graph->getPos(node->nodeId).X()][_graph->getPos(node->nodeId).Y()]++;、
        // if (node->net->getName() == "net_118568" || node->net->getName() == "net_116700")
        // if (RouteGraph::debugging)
        //     std::cout << node->net->getName() << ' ' << _graph->getVertexByIdx(node->nodeId)->getName() << " isDeleting: " << isDeleting << std::endl;
        // if (node->nodeId == 20514401) {
        //     std::cout << "ripupDfsSearch " << node->nodeId << ' ' << node->nodeDelay << std::endl;  
        // }
        if (_graph->getVertexCap(node->nodeId) < 0) {
	    //node->net->setRouteStatus(CONGESTED);
            if (node->net->useGlobalResult() && expanding) {
                int posX = _graph->getPos(node->nodeId).X(), posY = _graph->getPos(node->nodeId).Y();
                for (int i = -1; i <= 1; i++)
                    for (int j = -1; j <= 1; j++) {
                        if (!i && !j) continue;
                        if (posX + i < 0 || posX + i >= _graph->getWidth()) continue;
                        if (posY + j < 0 || posY + j >= _graph->getHeight()) continue;
                        node->net->addGlobalRouteResult(posX + i, posY + j);
                    }
            }
            // _graph->addVertexCost(node->nodeId, congestAddCost);
        }
        if (node->net->isSink(node->nodeId) && Pathfinder::isTimingDriven) {
            int posX = _graph->getPos(node->nodeId).X(), posY = _graph->getPos(node->nodeId).Y();
            int sourceX = _graph->getPos(node->net->getSource()).X(), sourceY = _graph->getPos(node->net->getSource()).Y();
            int dx = abs(posX - sourceX), dy = abs(posY - sourceY);
            COST_T costX = (dx / 6) * 120 + ((dx % 6) / 2) * 60 + (dx % 2) * 50;
            COST_T costY = (dy / 6) * 120 + ((dy % 6) / 2) * 60 + (dy % 2) * 50;
            COST_T delayCost = costX + costY + 100;
            if (node->nodeDelay > delayCost * 3 && Pathfinder::iter < 100 && _graph->getVertexSlack(node->nodeId) < -1000)
                isDeleting = true;
            if (node->nodeDelay > delayCost * 1.5 && Pathfinder::iter < 100 && _graph->getVertexSlack(node->nodeId) < -InstList::period)
                isDeleting = true;
            if (node->nodeDelay > delayCost * 5 && node->nodeDelay > InstList::period)
                isDeleting = true;
        }
        if (isDeleting || _graph->getVertexCap(node->nodeId) < 0 ) {
            isDeleting = res = true;
            for (auto child = node->firstChild; child != nullptr; child = child->right) {
                res |= ripupDfsSearch(child, true, expanding);
            }
            // if (_graph->getVertexCap(node->nodeId) <= 0)
            // _graph->addVertexCost(node->nodeId, -usedAddCost);
            eraseNode(node);
        }
        for (auto child = node->firstChild; child != nullptr; child = child->right)
            res |= ripupDfsSearch(child, isDeleting, expanding);
        // if (node->net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]" && node->nodeId == 35653)
        //     std::cout << node->firstChild << ' ' << isDeleting << std::endl;
        if (node->firstChild == nullptr && !node->net->isSink(node->nodeId) && !isDeleting && !(node->net->getSource() == node->nodeId))
            eraseNode(node);
        return res;
    }

    COST_T RouteTree::getTotalWL() {
        COST_T totalWL = 0;
        for (auto it : netRoot) {
            std::queue<std::shared_ptr<TreeNode>> q;
            q.push(it.second);
            while(!q.empty()) {
                auto now = q.front();
                q.pop();
                for (auto child = now->firstChild; child != nullptr; child = child->right) {
                    auto posS = _graph->getPos(now->nodeId);
                    auto posT = _graph->getPos(child->nodeId);
                    totalWL += abs(posT.X() - posS.X()) + abs(posT.Y() - posS.Y());
                    q.push(child);
                }
            }
        }
        return totalWL;
    }

}
