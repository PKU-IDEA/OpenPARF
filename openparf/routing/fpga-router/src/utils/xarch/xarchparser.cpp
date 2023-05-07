#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string.h>
#include <cmath>
#include <map>

#include "xarchnode.h"
#include "xarchparser.h"

XArchNodeType XArchParser::getNodeType(std::string type) {
    if (type == "DUMMY")
        return XArchNodeType::DUMMY;
    if (type == "LUT5")
        return XArchNodeType::LUT5;
    if (type == "LUT6")
        return XArchNodeType::LUT6;
    if (type == "LRAM")
        return XArchNodeType::LRAM;
    if (type == "SHIFT")
        return XArchNodeType::SHIFT;
    if (type == "DFF")
        return XArchNodeType::DFF;
    if (type == "CLA4")
        return XArchNodeType::CLA4;
    if (type == "INPAD")
        return XArchNodeType::INPAD;
    if (type == "OUTPAD")
        return XArchNodeType::OUTPAD;
    if (type == "GCU0")
        return XArchNodeType::GCU0;
    if (type == "BRAM36K")
        return XArchNodeType::BRAM36K;
    if (type == "RAMB")
        return XArchNodeType::RAMB;
    return XArchNodeType::UNDEFINED;
}

int XArchParser::getPinIdxInGraph(std::vector<std::vector<database::GridContent>>& layout, std::shared_ptr<RouteGraph> graph, XArchNode node, std::string nodePin) {
    int posx = node.posx, posy = node.posy, bel = node.bel;
    std::shared_ptr<database::Module> nodeModule;
    std::string nodePortName;
    int nodePinIndex = 0;

    if (nodePin.find("(") == std::string::npos) {
        nodePortName = nodePin;
        nodePinIndex = 0;
    } else {
        int strLen = nodePin.size();
        int pos    = nodePin.find("(");
        nodePortName = nodePin.substr(0, pos);
        nodePinIndex = std::stoi(nodePin.substr(pos + 1, strLen - pos - 2));
    }
    switch (node.type)
    {
    case XArchNodeType::UNDEFINED:
        std::cerr << "[Error] Undefined Node Found!" << std::endl;
        exit(1);
        break;
    case XArchNodeType::BRAM36K:
        nodeModule = layout[posx][posy].gridModule->getSubModule("RAMA[0]")->getSubModule("BRAM36K[0]");
        break;
    case XArchNodeType::CLA4:
        if(layout[posx][posy].gridModule->getName() == "FUAT")
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUA[0]")->getSubModule("CLA4[" + std::to_string(bel) + "]");
        else
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("CLA4[" + std::to_string(bel) + "]");
        break;
    case XArchNodeType::DFF:
        if(layout[posx][posy].gridModule->getName() == "FUAT")
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) +"]")->getSubModule("DFF[" + std::to_string(bel % 2) + "]");
        else
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("BLE[" + std::to_string(bel / 2) +"]")->getSubModule("DFF[" + std::to_string(bel % 2) + "]");
        break;
    case XArchNodeType::GCU0:
        nodeModule = layout[posx][posy].gridModule->getSubModule("RAMC[0]")->getSubModule("GCU0[0]");
        break;
    case XArchNodeType::INPAD:
        nodeModule = layout[posx][posy].gridModule->getSubModule("INPAD[" + std::to_string(bel) + "]");
        break;
    case XArchNodeType::OUTPAD:
        nodeModule = layout[posx][posy].gridModule->getSubModule("OUTPAD[" + std::to_string(bel) + "]");
        break;
    case XArchNodeType::LRAM:
        nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LRAM[0]");
        break;
    case XArchNodeType::LUT6:
        if(layout[posx][posy].gridModule->getName() == "FUAT")
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT6[0]");
        else
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT6[0]");
        break;
    case XArchNodeType::RAMB:
        nodeModule = layout[posx][posy].gridModule->getSubModule("RAMB[0]");
        break;
    case XArchNodeType::SHIFT:
        nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("SHIFT[0]");
        break;
    default:
        if(layout[posx][posy].gridModule->getName() == "FUAT")
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT5["  + std::to_string(bel % 2) + "]" );
        else
            nodeModule = layout[posx][posy].gridModule->getSubModule("FUB[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT5["  + std::to_string(bel % 2) + "]" );

        break;
    }
    return graph->getVertexId(posx, posy, nodeModule->getPort(nodePortName)->getPinByIdx(nodePinIndex));

}



std::vector<std::shared_ptr<Net>> XArchParser::buildNetlist(const char* placefile,
                                               const char* netfile,
                                               const char* nodefile,
                                               std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib,
                                               std::vector<std::vector<database::GridContent> >& layout,
                                               std::shared_ptr<RouteGraph> graph) {
    std::vector<std::shared_ptr<Net>> ret;

    std::ifstream nodeStream;
    nodeStream.open(nodefile);
    std::unordered_map<std::string, XArchNode> allNodes;
    std::string nodeName, nodeType;
    while (nodeStream >> nodeName >> nodeType) {
        // std::cout << nodeName << ' ' << nodeType << std::endl;
        allNodes[nodeName] = XArchNode(getNodeType(nodeType));
    }
    nodeStream.close();

    std::ifstream placeStream;
    placeStream.open(placefile);
    int _x, _y, _bel;
    std::vector<std::vector<std::vector<int> > > secondLUTStartPos;
    secondLUTStartPos.resize(graph->getWidth());
    for (int i = 0; i < graph->getWidth(); i++) {
        secondLUTStartPos[i].resize(graph->getHeight());
        for (int j = 0; j < graph->getHeight(); j++)
            secondLUTStartPos[i][j].resize(8, 0);
    }
    while (placeStream >> nodeName >> _x >> _y >> _bel) {
        // if (nodeName == "sig_1338")
        // std::cout << nodeName << ' ' << _x << ' ' << _y << ' ' << _bel << std::endl;
        allNodes[nodeName].posx = _x;
        allNodes[nodeName].posy = _y;
        allNodes[nodeName].bel  = _bel;
        buildInst(layout, graph, allNodes[nodeName], nodeName);
        if (_bel % 2 == 1) continue;
        if (allNodes[nodeName].type == XArchNodeType::LUT5){
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 5;
        }
    }
    placeStream.close();

    std::ifstream netStream;
    netStream.open(netfile);
    std::string temp, netName;
    int netSize;
    while (netStream >> temp >> netName >> netSize) {
        std::map<std::pair<std::pair<INDEX_T, INDEX_T>, int>, std::string> BLEVis;
        // bool flag = 0;
        std::shared_ptr<Net> net = std::make_shared<Net>(netName);
        bool hasBUFGCE = false;
        std::vector<std::string> nodeNames;
        std::vector<std::string> nodePins;
        for (int i = 0; i < netSize; i++) {
            std::string nodeName, nodePin;
            netStream >> nodeName >> nodePin;
            if (nodeName == "VCC") continue;
            // if (allNodes[nodeName].type == NodeType::BUFGCE) {
            //     hasBUFGCE = true;
            // }
            // else {
            if (nodePin.find("SUB_DOUT") != std::string::npos || nodePin.find("CE") != std::string::npos) continue;
            nodeNames.push_back(nodeName);
            nodePins.push_back(nodePin);
            if (allNodes[nodeName].type == XArchNodeType::LUT5 && nodePin[0] == 'I') {
                INDEX_T posx = allNodes[nodeName].posx;
                INDEX_T posy = allNodes[nodeName].posy;
                INDEX_T bel = allNodes[nodeName].bel;
                if (bel % 2 == 0) {
                    // std::cout << posx << ' ' << posy << ' ' << bel << std::endl;
                    BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)] = nodePin;
                }
            }
            // }
            // if (nodePin[0] == 'I') lutIn.insert(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy));
            // if (nodePin[0] == 'D') ffIn.insert(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy));
        }
        // std::string nodeName, nodePin;
        // netStream >> nodeName >> nodePin;
        // if ((allNodes[nodeName].type == XArchNodeType::LUT5) && nodePin[0] == 'I') {
        //     INDEX_T posx = allNodes[nodeName].posx;
        //     INDEX_T posy = allNodes[nodeName].posy;
        //     INDEX_T bel = allNodes[nodeName].bel;
        //     if (bel % 2 == 0)
        //         BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)] = nodePin;
        // }
        netStream >> temp;
        // if (allNodes[nodeName].type == NodeType::BUFGCE) {
        //     hasBUFGCE = true;
        // }
        // if (hasBUFGCE) {
        //     continue;
        // }
        int nodeNameSize = nodeNames.size();
        // if (net->getName() == "net_33534") {
        //     std::cerr << "nodeNameSize: " << nodeNameSize << std::endl;
        // }
        int sourceCnt = 0;
        for (int i = 0; i < nodeNameSize; i++) {
            auto& nodeName = nodeNames[i];
            std::string nodePin;
            INDEX_T posx = allNodes[nodeName].posx;
            INDEX_T posy = allNodes[nodeName].posy;
            INDEX_T bel = allNodes[nodeName].bel;
            int select = -1;

            if (allNodes[nodeName].type != XArchNodeType::LUT5) nodePin = nodePins[i];
            else if (bel % 2 == 1 && BLEVis.find(std::make_pair(std::make_pair(posx, posy), bel / 2)) != BLEVis.end()) {
                select = 0;
                nodePin = BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)];
                // std::cout << nodePin << std::endl;
                // std::cout << posx << ' ' << posy << ' ' << bel << ' ' << BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)] << std::endl;
            }
            else if (bel % 2 == 1 && secondLUTStartPos[posx][posy][bel / 2]) {
                select = 1;
                nodePin = "IN(X)";
                nodePin[3] = '0' + secondLUTStartPos[posx][posy][bel / 2];
                secondLUTStartPos[posx][posy][bel / 2]++;
                // std::cout << nodePin << std::endl;
            }
            else {
                select = 2;
                nodePin = nodePins[i];
                // std::cout << nodePin << std::endl;
            }
            // std::cout << select << ' ' << nodePin << std::endl;
            int pinIdx = getPinIdxInGraph(layout, graph, allNodes[nodeName], nodePin);
            assert(graph->getVertexInst(pinIdx) != -1);
            if (graph->getVertexByIdx(pinIdx)->getPinPort()->getPortType() == database::PortType::INPUT) {
                net->addSink(pinIdx);
            }
            else {
                sourceCnt++;
                net->setSource(pinIdx);
            }
            net->addGuideNode(allNodes[nodeName].posx, allNodes[nodeName].posy);
            net->addGuideNode(posx + layout[posx][posy].width - 1, posy + layout[posx][posy].height - 1);

        }
        // net->setSource(getPinIdxInGraph(layout, graph, allNodes[nodeName], nodePin));
        // net->addGuideNode(allNodes[nodeName].posx, allNodes[nodeName].posy);
        // INDEX_T posx = allNodes[nodeName].posx;
        // INDEX_T posy = allNodes[nodeName].posy;
        // net->addGuideNode(posx + layout[posx][posy].width - 1, posy + layout[posx][posy].height - 1);
        if (sourceCnt == 1 && netSize < 3000 && net->getSinkSize()) {
            ret.push_back(net);
            int source = net->getSource();
            assert(graph->getVertexInst(source) != -1);
            for (auto sink : net->getSinks()) {
                assert(graph->getVertexInst(sink) != -1);
                graph->getInstList().getInsts()[graph->getVertexInst(sink)].addInputNetNum();
                graph->getInstList().getInsts()[graph->getVertexInst(source)].addOutputNetNum();
            }
        }
        if (sourceCnt > 1) {
            std::cout << net->getName() << std::endl;
            exit(1);
        }
    }
    netStream.close();
    std::cout << "Netlist Size: " << ret.size() << std::endl;
    return std::move(ret);
}