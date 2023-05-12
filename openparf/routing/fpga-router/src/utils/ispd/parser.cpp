#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string.h>
#include <cmath>
#include <map>

#include "ispdnode.h"
#include "parser.h"
NodeType getNodeType(std::string type) {
    if (type == "FDRE")
        return NodeType::FDRE;
    if (type == "LUT6")
        return NodeType::LUT6;
    if (type == "LUT5")
        return NodeType::LUT5;
    if (type == "LUT4")
        return NodeType::LUT4;
    if (type == "LUT3")
        return NodeType::LUT3;
    if (type == "LUT2")
        return NodeType::LUT2;
    if (type == "CARRY8")
        return NodeType::CARRY8;
    if (type == "DSP48E2")
        return NodeType::DSP48E2;
    if (type == "RAMB36E2")
        return NodeType::RAMB36E2;
    if (type == "BUFGCE")
        return NodeType::BUFGCE;
    if (type == "IBUF")
        return NodeType::IBUF;
    if (type == "OBUF")
        return NodeType::OBUF;
    return NodeType::UNDEFINE;
}

int getPinIdxInGraph(std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib, std::shared_ptr<RouteGraph> graph, ISPDNode node, std::string nodePin) {
    int posx = node.posx, posy = node.posy, bel = node.bel;
    std::shared_ptr<database::Module> nodeModule;
    std::string nodePortName;
    int nodePinIndex;
    if (nodePin.find("[") == std::string::npos) {
        nodePortName = nodePin;
        nodePinIndex = 0;
    } else {
        int strLen = nodePin.size();
        int pos    = nodePin.find("[");
        nodePortName = nodePin.substr(0, pos);
        nodePinIndex = std::stoi(nodePin.substr(pos + 1, strLen - pos - 2));
    }
    switch (node.type)
    {
    case NodeType::UNDEFINE:
        std::cerr << "[Error] Undefined Node Found!" << std::endl;
        exit(1);
        break;
    case NodeType::FDRE:
        nodeModule = lib["FUAT"]->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) +"]")->getSubModule("DFF[" + std::to_string(bel % 2) + "]");
        break;
    case NodeType::LUT6:
        nodeModule = lib["FUAT"]->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT6[0]");
        break;
    case NodeType::CARRY8:
        std::cerr << "[Error] CARRY8 is not support in this arch" << std::endl;
        exit(1);
        break;
    case NodeType::DSP48E2:
        nodeModule = lib["RAMBT"]->getSubModule("RAMB[0]")->getSubModule("DSP48[0]");
        break;
    case NodeType::RAMB36E2:
        nodeModule = lib["RAMAT"]->getSubModule("RAMA[0]")->getSubModule("BRAM36K[0]");
        break;
    case NodeType::IBUF:
        nodeModule = lib["IO"]->getSubModule("inpad[" + std::to_string(bel) + "]");
        break;
    case NodeType::OBUF:
        nodeModule = lib["IO"]->getSubModule("outpad[" + std::to_string(bel) + "]");
        break;
    case NodeType::BUFGCE:
        nodeModule = lib["IO"]->getSubModule("bufgce[" + std::to_string(bel) + "]");
        break;
    default:
        nodeModule = lib["FUAT"]->getSubModule("FUA[0]")->getSubModule("BLE[" + std::to_string(bel / 2) + "]")->getSubModule("LUT[0]")->getSubModule("LUT5["  + std::to_string(bel % 2) + "]" );
        break;
    }
    return graph->getVertexId(posx, posy, nodeModule->getPort(nodePortName)->getPinByIdx(nodePinIndex));
}

void shiftNodePos(int &x, int &y) {
    const int DSPPOS[] = {29, 66, 104, 142};
    for (int i = 0; i < 4; i++) {
        if (x > DSPPOS[i])
            x++;
        if (x == DSPPOS[i] && y%5 != 0) {
            x++;
            y = y - y%5;
        }
    }

}


std::vector<std::shared_ptr<Net>> buildNetlist(const char*                  placefile,
        const char*                                                         netfile,
        const char*                                                         nodefile,
        std::unordered_map<std::string, std::shared_ptr<database::Module>>& lib,
        std::vector<std::vector<database::GridContent>>&                    layout,
        std::shared_ptr<RouteGraph>                                         graph) {
    std::vector<std::shared_ptr<Net>> ret;

    std::ifstream                     nodeStream;
    nodeStream.open(nodefile);
    std::unordered_map<std::string, ISPDNode> allNodes;
    std::string                               nodeName, nodeType;
    while (nodeStream >> nodeName >> nodeType) {
        // std::cout << nodeName << ' ' << nodeType << std::endl;
        allNodes[nodeName] = ISPDNode(getNodeType(nodeType));
    }
    nodeStream.close();
    std::ifstream placeStream;
    placeStream.open(placefile);
    int                                        _x, _y, _bel;
    std::vector<std::vector<std::vector<int>>> secondLUTStartPos;
    secondLUTStartPos.resize(graph->getWidth());
    for (int i = 0; i < graph->getWidth(); i++) {
        secondLUTStartPos[i].resize(graph->getHeight());
        for (int j = 0; j < graph->getHeight(); j++) secondLUTStartPos[i][j].resize(8, 0);
    }
    std::string line;
    while (std::getline(placeStream, line)) {

        // ignore the "FIXED" keyword at the end of the line
        std::size_t pos = line.find("FIXED");
        if (pos != std::string::npos) {
            line.erase(pos);
        }

        std::istringstream iss(line);
        iss >> nodeName >> _x >> _y >> _bel;

        if (allNodes[nodeName].type == NodeType::IBUF || allNodes[nodeName].type == NodeType::OBUF) {
            _y += _bel / 8;
            _bel %= 8;
        }
        shiftNodePos(_x, _y);
        // std::cout << nodeName << ' ' << _x << ' ' << _y << ' ' << _bel << std::endl;
        allNodes[nodeName].posx = _x;
        allNodes[nodeName].posy = _y;
        allNodes[nodeName].bel  = _bel;
        if (_bel % 2 == 1) continue;
        if (allNodes[nodeName].type == NodeType::LUT1) {
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 1;
        }
        if (allNodes[nodeName].type == NodeType::LUT2) {
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 2;
        }
        if (allNodes[nodeName].type == NodeType::LUT3) {
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 3;
        }
        if (allNodes[nodeName].type == NodeType::LUT4) {
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 4;
        }
        if (allNodes[nodeName].type == NodeType::LUT5) {
            // std::cout << _x << ' ' << _y << ' ' << _bel << std::endl;
            secondLUTStartPos[_x][_y][_bel / 2] = 5;
        }
    }
    placeStream.close();
    std::ifstream netStream;
    netStream.open(netfile);
    std::string temp, netName;
    int         netSize;
    // int fuBypUsePredict[180][485];
    // memset(fuBypUsePredict, 0, sizeof(fuBypUsePredict));
    // std::set<std::pair<int, int>> lutIn, ffIn;
    while (netStream >> temp >> netName >> netSize) {
        std::map<std::pair<std::pair<INDEX_T, INDEX_T>, int>, std::string> BLEVis;
        // bool flag = 0;
        std::shared_ptr<Net>                                               net(new Net(netName));
        bool                                                               hasBUFGCE = false;
        std::vector<std::string>                                           nodeNames;
        std::vector<std::string>                                           nodePins;
        for (int i = 0; i < netSize; i++) {
            std::string nodeName, nodePin;
            netStream >> nodeName >> nodePin;
            if (allNodes[nodeName].type == NodeType::BUFGCE) {
              hasBUFGCE = true;
            } else {
              nodeNames.push_back(nodeName);
              nodePins.push_back(nodePin);
              if (allNodes[nodeName].type == NodeType::LUT1 || allNodes[nodeName].type == NodeType::LUT2 ||
                      allNodes[nodeName].type == NodeType::LUT3 || allNodes[nodeName].type == NodeType::LUT4 ||
                      allNodes[nodeName].type == NodeType::LUT5) {
                INDEX_T posx = allNodes[nodeName].posx;
                INDEX_T posy = allNodes[nodeName].posy;
                INDEX_T bel  = allNodes[nodeName].bel;
                if (bel % 2 == 0 && nodePin[0] == 'I') {
                  // std::cout << posx << ' ' << posy << ' ' << bel << std::endl;
                  BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)] = nodePin;
                }
              }
              // if (net->getName() == "net_100193") {
              //     std::cout << nodePin << ' ' << nodePins[i] << std::endl;
              //     getchar();
              // }
            }
            // if (nodePin[0] == 'I') lutIn.insert(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy));
            // if (nodePin[0] == 'D') ffIn.insert(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy));
        }
        // std::string nodeName, nodePin;
        // netStream >> nodeName >> nodePin;
        // if ((allNodes[nodeName].type == NodeType::LUT1 || allNodes[nodeName].type == NodeType::LUT2 ||
        // allNodes[nodeName].type == NodeType::LUT3 || allNodes[nodeName].type == NodeType::LUT4 ||
        // allNodes[nodeName].type == NodeType::LUT5) && nodePin.find("I") != std::string::npos) {
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
        if (hasBUFGCE) {
            continue;
        }
        int nodeNameSize = nodeNames.size();
        // if (net->getName() == "net_33534") {
        //     std::cerr << "nodeNameSize: " << nodeNameSize << std::endl;
        // }
        int outputCnt    = 0;
        for (int i = 0; i < nodeNameSize; i++) {
            auto&       nodeName = nodeNames[i];
            std::string nodePin;
            INDEX_T     posx   = allNodes[nodeName].posx;
            INDEX_T     posy   = allNodes[nodeName].posy;
            INDEX_T     bel    = allNodes[nodeName].bel;
            int         select = -1;
            int         typeId = (int) allNodes[nodeName].type;
            if (typeId < 2 || typeId > 6) nodePin = nodePins[i];
            else if (bel % 2 == 1 && BLEVis.find(std::make_pair(std::make_pair(posx, posy), bel / 2)) != BLEVis.end() &&
                     nodePins[i][0] == 'I') {
              select  = 0;
              nodePin = BLEVis[std::make_pair(std::make_pair(posx, posy), bel / 2)];
              // std::cout << nodePin << std::endl;
              // std::cout << posx << ' ' << posy << ' ' << bel << ' ' << BLEVis[std::make_pair(std::make_pair(posx,
              // posy), bel / 2)] << std::endl;
            } else if (bel % 2 == 1 && secondLUTStartPos[posx][posy][bel / 2] && nodePins[i][0] == 'I') {
              select     = 1;
              nodePin    = "IX";
              nodePin[1] = '0' + secondLUTStartPos[posx][posy][bel / 2];
              secondLUTStartPos[posx][posy][bel / 2]++;
              // std::cout << nodePin << std::endl;
            } else {
              select  = 2;
              nodePin = nodePins[i];
              // std::cout << nodePin << std::endl;
            }
            if (nodePin == "RSTRAMARSTRAM") continue;
            if (nodePin == "RSTRAMB") continue;
            if (nodePin == "RSTREGARSTREG") continue;
            if (nodePin == "RSTREGB") continue;
            if (nodePin == "CLKARDCLK") continue;
            if (nodePin == "CLKBWRCLK") continue;
            if (nodePin == "CE") continue;
            if (nodePin == "R") continue;
            int pinIdx = getPinIdxInGraph(lib, graph, allNodes[nodeName], nodePin);
            // std::cout << select << ' ' << nodePin << std::endl;
            if (graph->getVertexByIdx(pinIdx)->getPinPort()->getPortType() == database::PortType::INPUT) {
              net->addSink(pinIdx);
              net->addGuideNode(allNodes[nodeName].posx, allNodes[nodeName].posy);
              // if (fabs(posx - 100) <= 2 && fabs(posy - 374) <= 2) flag = 1;
              net->addGuideNode(posx + layout[posx][posy].width - 1, posy + layout[posx][posy].height - 1);
            } else {
              // if (net->getName() == "net_100193") {
              //     std::cout << nodePin << ' ' << nodePins[i] << ' ' << graph->getVertexByIdx(pinIdx)->getName() <<
              //     std::endl; getchar();
              // }
              net->setSource(getPinIdxInGraph(lib, graph, allNodes[nodeName], nodePin));
              net->addGuideNode(allNodes[nodeName].posx, allNodes[nodeName].posy);
              net->addGuideNode(posx + layout[posx][posy].width - 1, posy + layout[posx][posy].height - 1);
              outputCnt++;
            }
        }
        // net->setSource(getPinIdxInGraph(lib, graph, allNodes[nodeName], nodePin));
        // net->addGuideNode(allNodes[nodeName].posx, allNodes[nodeName].posy);
        // INDEX_T posx = allNodes[nodeName].posx;
        // INDEX_T posy = allNodes[nodeName].posy;
        // // if (fabs(posx - 100) <= 2 && fabs(posy - 374) <= 2) flag = 1;
        // net->addGuideNode(posx + layout[posx][posy].width - 1, posy + layout[posx][posy].height - 1);
        // int threshold = 8;
        // if (nodePin[0] == 'O') {
        //     if (lutIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != lutIn.end() &&
        //     fuBypUsePredict[allNodes[nodeName].posx][allNodes[nodeName].posy] >= threshold)
        //         continue;
        //     fuBypUsePredict[allNodes[nodeName].posx][allNodes[nodeName].posy] +=
        //     (lutIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != lutIn.end());

        // }
        // if (nodePin[0] == 'Q') {
        //     if ((lutIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != lutIn.end() ||
        //     ffIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != ffIn.end()) &&
        //     fuBypUsePredict[allNodes[nodeName].posx][allNodes[nodeName].posy] >= threshold)
        //         continue;
        //     fuBypUsePredict[allNodes[nodeName].posx][allNodes[nodeName].posy]
        //     += (lutIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != lutIn.end() ||
        //     ffIn.find(std::make_pair(allNodes[nodeName].posx, allNodes[nodeName].posy)) != ffIn.end());
        // }
        // if (flag == 1)
        if (outputCnt == 1) ret.push_back(net);
        else {
            std::cerr << net->getName() << ' ' << outputCnt << std::endl;
        }
    }
    netStream.close();
    std::cout << "Netlist Size: " << ret.size() << std::endl;
    // exit(-1);

    // for (int i = 0; i < 172; i++)
    //     for (int j = 0; j < 480; j++) {
    //         if (fuBypUsePredict[i][j] > 16)
    //             std::cout << "[Warning] o_fu_byp maybe not enough for grid " << i << ' ' << j << " Required to
    //             use: " << fuBypUsePredict[i][j] << std::endl;
    //     }
    // exit(0);
    return std::move(ret);
}