#include "globalrouteresult.h"

#include <fstream>
#include <iostream>
#include <unordered_map>
namespace router {
    void printGlobalRouteResult(std::vector<std::shared_ptr<Net>>& netlist, std::string fileName) {
        
        std::ofstream ofs(fileName);
        for (auto net : netlist) {
            ofs << net->getName() << ' ' << net->getGlobalRouteResult().size() << std::endl;
            auto &gr = net->getGlobalRouteResult();
            for (auto node : gr) {
                ofs << node.first << ' ' << node.second << std::endl;
            }
        }
        ofs.close();
    }

    void loadGlobalRouteResult(std::vector<std::shared_ptr<Net>>& netlist, std::string fileName) {
        std::unordered_map<std::string, std::shared_ptr<Net>> netNameMap;
        std::ifstream ifs(fileName);
        for (auto net : netlist) {
            netNameMap[net->getName()] = net; 
        }
        std::string netName;
        int grSize;
        while (ifs >> netName >> grSize) {
            auto net = netNameMap[netName];            
            for (int i = 0; i < grSize; i++) {
                INDEX_T x, y;
                ifs >> x >> y;
                net->addGlobalRouteResult(x, y);
            }
        }
        ifs.clear();
    }
};