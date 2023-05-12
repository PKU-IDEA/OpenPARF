#include "printer.h"
#include "router/routetree.h"
#include "router/pathfinder.h"
#include <limits>

namespace router {

void printRouteResult(std::vector<std::shared_ptr<router::Net>>& netlist, std::string fileName, std::shared_ptr<router::RouteGraph> graph) {
    using std::max;
    using std::min;
    // Pathfinder::routetree.ripup();

    pugi::xml_document file;
    pugi::xml_node netlistInfo = file.append_child("netlist");
    for (auto net : netlist) {
        // std::cout << net->getName() << std::endl;
        pugi::xml_node netInfo = netlistInfo.append_child("net");
        netInfo.append_attribute("name") = net->getName().c_str();
        pugi::xml_node sourceInfo = netInfo.append_child("source");
        sourceInfo.append_attribute("pin") = graph->getVertexByIdx(net->getSource())->getName().c_str();
        sourceInfo.append_attribute("x") = graph->getPos(net->getSource()).X();
        sourceInfo.append_attribute("y") = graph->getPos(net->getSource()).Y();
        int x, y;
        if (net->getRouteStatus() == router::RouteStatus::FAILED) {
                netInfo.append_attribute("result") = "Failed";
                // continue;
        }
        else if (net->getRouteStatus() == router::RouteStatus::CONGESTED) {
                netInfo.append_attribute("result") = "Congested";
        }
        else 
        netInfo.append_attribute("result") = (net->getRouteStatus() == router::RouteStatus::SUCCESS ? "Success" : "Unrouted");
        if (net->getRouteStatus() == router::RouteStatus::FAILED) {
            auto& gr = net->getGlobalRouteResult();
            for (auto pos : gr) {
                auto grInfo = netInfo.append_child("global_result");
                grInfo.append_attribute("x") = pos.first;
                grInfo.append_attribute("y") = pos.second;
            }
        }
        
        router::RouteTree& routetree = router::Pathfinder::routetree;
        int sinkSize = net->getSinkSize();
        for (int i = 0; i < sinkSize; i++) {
            // std::cout << "Sink #" << i << std::endl;
            // std::cout << "STEP 0" << std::endl;
            pugi::xml_node sinkInfo = netInfo.append_child("sink");
            int pinId = net->getSinkByIdx(i);
            sinkInfo.append_attribute("pin") = graph->getVertexByIdx(pinId)->getName().c_str();
            sinkInfo.append_attribute("x") = graph->getPos(pinId).X();
            sinkInfo.append_attribute("y") = graph->getPos(pinId).Y();
            // std::cout << "STEP 1" << std::endl;
            

            if (net->getRouteStatus() != SUCCESS) continue;
            if (pinId >= graph->getVertexNum()) {
                std::cerr << "[Fatal Error] pinId is larger than pin Num" << std::endl;
                exit(1);
            }
            std::shared_ptr<TreeNode> node = routetree.getTreeNodeByIdx(pinId);
            if (pinId == 20514401)
            std::cout << "printRouteResult: " << pinId << ' ' << node->nodeDelay << std::endl;
            sinkInfo.append_attribute("delay") = node->nodeDelay;
            sinkInfo.append_attribute("nodeid") = node->nodeId;
            if (node == nullptr) {
                std::cerr << "[Fatal Error] Nullptr found at Net " << net->getName() << " leaf node " << graph->getVertexByIdx(pinId)->getName() << "!" << std::endl;
                exit(1);
            }
            // std::cout << "STEP 2" << std::endl;
            while(node->father != nullptr) {
                int prevId = node->father->nodeId;
                // std::cout << node->nodeId << ' ' << node->net->getName() << ' ' << graph->getVertexByIdx(node->nodeId)->getName() << std::endl;
                if (prevId >= graph->getVertexNum()) {
                    std::cerr << "[Fatal Error] prevId is larger than pin Num" << std::endl;
                    std::cerr << "prevId: " << prevId << " nodeId: " << node->nodeId << std::endl; 
                    exit(1);
                }
                // std::cout << "pinId: " << pinId << " prevId: " << prevId << std::endl;
                std::shared_ptr<database::Pin> prevPin = graph->getVertexByIdx(prevId);
                pugi::xml_node pathInfo = sinkInfo.append_child("path");
                pathInfo.append_attribute("input") = prevPin->getName().c_str();
                pathInfo.append_attribute("x") = graph->getPos(prevId).X();
                pathInfo.append_attribute("y") = graph->getPos(prevId).Y();
                pathInfo.append_attribute("delay") = node->father->nodeDelay;
                pathInfo.append_attribute("nodeid") = node->father->nodeId;
                if (graph->getVertexCap(prevId) < 0)
                    pathInfo.append_attribute("congest_with_net") = routetree.getTreeNodeByIdx(prevId)->net->getName().c_str();
                node = node->father;
            }
            // std::cout << "STEP 3" << std::endl;
            // getPinGrid(net->getSinkByIdx(i)->getName(),  x, y);
        }
    }
    file.save_file(fileName.c_str());     
}

void getPinGrid(const std::string &pinName, int &x, int &y) {
    int pos = pinName.find("][");
    if (pos == std::string::npos) {
        x = -1, y = -1;
        return;
    }
    else {
        x = 0;
        int pow_num = 1;
        for (int i = pos - 1; i >= 0 && isdigit(pinName[i]); i--) {
            x += pow_num * (pinName[i] - '0');
            pow_num *= 10;
        }
        pow_num = 1;
        y = 0;
        for (int j = pos + 2; j < pinName.size() && isdigit(pinName[j]); j++) {
            y = y * pow_num + (pinName[j] - '0');
            pow_num *= 10;
        }
    }
}

} // namespace router