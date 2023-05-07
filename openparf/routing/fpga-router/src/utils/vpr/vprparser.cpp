#include "vprparser.h"
#include <pugixml/pugixml.hpp>

#include <iostream>
#include <fstream>
#include <queue>
#include <set>
namespace router {
    VertexType getVertexType(std::string typeName) {
        if (typeName == "OPIN") return OPIN;
        if (typeName == "IPIN") return IPIN;
        if (typeName == "SOURCE") return SOURCE;
        if (typeName == "SINK") return SINK;
        if (typeName == "CHANX") return CHANX;
        if (typeName == "CHANY") return CHANY;
        return NONE;
    }
    void printVPRWirelength(std::vector<std::shared_ptr<Net>>& netlist, std::shared_ptr<RouteGraph> graph) {
        COST_T totalWL = 0;
        for (auto net : netlist) {
            //  if (net->getName() != "sparc_exu:exu|bw_r_irf:irf|bw_r_irf_core:bw_r_irf_core|bw_r_irf_register:register16|window_rtl_0_bypass[1]")
            //     continue;

            std::queue<std::shared_ptr<TreeNode>> q;
            q.push(Pathfinder::routetree.getNetRoot(net));
            while (!q.empty()) {
                auto now = q.front();
                q.pop();

                auto type = graph->getVertexType(now->nodeId);
                auto pos = graph->getPos(now->nodeId);
                auto posHigh = graph->getPosHigh(now->nodeId);
                // std::cout << now->nodeId << ' ' << (int)type << ' ' << pos.X() << ' ' << pos.Y() << ' ' << posHigh.X() << ' ' << posHigh.Y() << std::endl;
                if (type == CHANX || type == CHANY) {
                    totalWL += 1 + posHigh.X() - pos.X() + posHigh.Y() - pos.Y();
                }
                for (auto child = now->firstChild; child != nullptr; child = child->right) {
                    q.push(child);
                }
            }
            // std::cout << "Net :" << net->getName() << ' ' << " Sinks: " << net->getSinkSize() << " totalWL: " << totalWL << std::endl;
        }

        std::cout << "total Wire length: " << totalWL << std::endl;
    }

    std::shared_ptr<RouteGraph> parseRRGraph(const char* fileName) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(fileName);

        auto graphInfo = doc.child("rr_graph");
        auto gridInfo = graphInfo.child("grid");
        int width = 0, height = 0;
        for (auto locInfo : gridInfo.children("grid_loc")) {
            width = std::max(width, locInfo.attribute("x").as_int() + 1);
            height = std::max(width, locInfo.attribute("y").as_int() + 1);
        }
        auto nodesInfo = graphInfo.child("rr_nodes");
        std::shared_ptr<RouteGraph> graph = std::make_shared<RouteGraph>(width, height, 0);
        std::shared_ptr<GlobalRouteGraph> globalGraph = graph->getGlobalGraph();
        for (auto nodeInfo : nodesInfo.children("node")) {
            auto locInfo = nodeInfo.child("loc");
            int posX = locInfo.attribute("xlow").as_int();
            int posY = locInfo.attribute("ylow").as_int();
            int posXH = locInfo.attribute("xhigh").as_int();
            int posYH = locInfo.attribute("yhigh").as_int();
            int vertexIdx = graph->addVertex(posX, posY, posXH, posYH, nodeInfo.attribute("capacity").as_int(), getVertexType(nodeInfo.attribute("type").value()));
            // for (int i = posX; i <= posXH; i++)
            //     for (int j = posY; j <= posYH; j++) {
            //         if (i != posXH) {
            //             globalGraph->addEdge(i, j, i + 1, j);
            //             globalGraph->addEdge(i + 1, j, i, j);
            //         }
            //         if (j != posYH) {
            //             globalGraph->addEdge(i, j, i, j + 1);
            //             globalGraph->addEdge(i, j + 1, i, j);
            //         }
            //     }
            // if (posX != posXH || posY != posYH)
            // std::cout << "vertexIdx " << vertexIdx << " posX " << posX << ' ' << " posY " << posY << " posXH " << posXH << " posYH " << posYH << std::endl;
        }

        auto edgesInfo = graphInfo.child("rr_edges");
        for (auto edgeInfo : edgesInfo.children("edge")) {
            int src = edgeInfo.attribute("src_node").as_int();
            int sink = edgeInfo.attribute("sink_node").as_int();
            int srcX = graph->getPos(src).X(), srcY = graph->getPos(src).Y();
            int srcXH = graph->getPosHigh(src).X(), srcYH = graph->getPosHigh(src).Y();
            int sinkX = graph->getPos(sink).X(), sinkY = graph->getPos(sink).Y();
            int sinkXH = graph->getPosHigh(sink).X(), sinkYH = graph->getPosHigh(sink).Y();
            // globalGraph->addEdge(srcX, srcY, sinkX, sinkY);
            COST_T edgeCost;
            if (graph->getVertexType(sink) == SOURCE)
                edgeCost = 1.0 * baseCost;
            else if (graph->getVertexType(sink) == SINK)
                edgeCost = 0 * baseCost;
            else if (graph->getVertexType(sink) == OPIN)
                edgeCost = 1.0 * baseCost;
            else if (graph->getVertexType(sink) == IPIN)
                edgeCost = 0.95 * baseCost;
            if (graph->getVertexType(sink) == CHANX || graph->getVertexType(sink) == CHANY)
                edgeCost = (1 + (sinkXH - sinkX) + (sinkYH - sinkY)) * baseCost;
            graph->addEdge(src, sink, edgeCost);
        }
        // graph->globalGraph = std::make_shared<GlobalRouteGraph> (width, height);
        return graph;
    }

    std::vector<std::shared_ptr<Net>> parseRouteFile(const char* fileName, std::shared_ptr<RouteGraph> graph) {
        std::vector<std::shared_ptr<Net>> res;
        std::ifstream ifs(fileName);

        std::string buffer;
        bool is_global_net = false;
        std::shared_ptr<Net> currentNet;
        while (getline(ifs, buffer)) {
            // if (currentNet != nullptr)
            // std::cout << currentNet->getName() << std::endl;
            if (buffer.find("Net ") != std::string::npos) {
                if (buffer.find("global net connecting") != std::string::npos) {
                    is_global_net = true;
                    continue;
                }
                else {
                    is_global_net = false;
                    int pos_s = buffer.find("(");
                    int pos_t = buffer.find(")");
                    std::string netName = buffer.substr(pos_s + 1, pos_t - pos_s - 1);
                    currentNet = std::make_shared<Net>(netName);
                    res.push_back(currentNet);
                }
            }
            else {
                if (is_global_net) continue;
                if (buffer.find("Node") != std::string::npos) {
                    int length = buffer.size();
                    bool is_parsing = false;
                    int nodeId = 0;
                    for (int i = 0; i < length; i++) {
                        if (isdigit(buffer[i]) && !is_parsing) {
                            is_parsing = true;
                        }
                        if (!isdigit(buffer[i]) && is_parsing)
                            break;
                        if (is_parsing) nodeId = nodeId * 10 + (buffer[i] - '0');
                    }
                    if (buffer.find("SOURCE") != std::string::npos) {
                        currentNet->setSource(nodeId);
                        // std::cout << nodeId << ' ' << graph->getPos(nodeId).X() << ' ' << graph->getPos(nodeId).Y() << std::endl;
                    }
                    if (buffer.find("SINK") != std::string::npos) {
                        currentNet->addSink(nodeId);
                        // std::cout << nodeId << ' ' << graph->getPos(nodeId).X() << ' ' << graph->getPos(nodeId).Y() << std::endl;
                    }
                    // std::cout << graph->getPosHigh(nodeId).X() << ' ' << graph->getPosHigh(nodeId).Y() << std::endl;
                    currentNet->addGuideNode(graph->getPos(nodeId).X(), graph->getPos(nodeId).Y());
                    currentNet->addGuideNode(graph->getPosHigh(nodeId).X(), graph->getPosHigh(nodeId).Y());
                }
            }
        }

        return std::move(res);
    }
} // namespace router
