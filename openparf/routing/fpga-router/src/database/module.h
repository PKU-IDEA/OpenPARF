#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>

#include "port.h"
#include "grid.h"

namespace database {


class Module {
public: 
    Module(){}
    ~Module(){
        // std::cout << "Deleting Module " + name << std::endl; 
        // ports.clear();
        // submodules.clear();
        // subGrid = nullptr;
    }
    Module(std::string _name);

    std::string getName() { return name; }
    std::shared_ptr<Grid> getSubGrid() { return subGrid; }

    std::shared_ptr<Port> addPort(std::string portName, PortType portType, int width, std::string portInfo);
    std::shared_ptr<Port> addPort(std::string portName, PortType portType, int width, std::string portInfo, int x, int y);
    void addSubModule(std::string subModuleName, std::shared_ptr<Module> subModule);
    void setSubGrid(std::string gridName, int width, int height);
    void setSubModuleNum(std::string subModuleName, int num) { subModuleNum[subModuleName] = num; }

    std::shared_ptr<Module> getSubModule(std::string _name) { 
        if (submodules.find(_name) == submodules.end()) return nullptr;
        return submodules[_name];     
    }
    std::shared_ptr<Port> getPort(std::string portName) {
        if (ports.find(portName) == ports.end()) return nullptr;
        return ports[portName];
    }
    int getSubModuleNum(std::string subModuleName) { 
        if (subModuleNum.find(subModuleName) == subModuleNum.end()) return 0;
        return subModuleNum[subModuleName]; 
    }
    
    std::unordered_map<std::string, std::shared_ptr<Port> >& allPorts() { return ports; }
    std::unordered_map<std::string, std::shared_ptr<Module> >& allSubmodules() { return submodules; } 

    void listSubModules() {
        for (auto it : submodules) {
            std::cout << it.first << std::endl;
        }
    }
    
    static std::unordered_map<std::string, int> GSWInterConnectLength;

    void addMode(std::string modeName) { modeNames.push_back(modeName); }
    int getModeSelect() { return modeSelect; }
    int selectMode(int mode) { modeSelect = mode; }
    int setModeBelong(int mode) { modeBelong = mode; }
    int getModeBelong() {return modeBelong; }

    void addGSW(std::shared_ptr<Module> gsw) { global_sw.push_back(gsw); }
    std::shared_ptr<Module> getGSWbyIdx(int id) { return global_sw[id]; }
    int getGSWSize() { return global_sw.size(); }

private:
    int modeSelect; 
    int modeBelong; // belongs to which mode of parent module

    std::string name;
    std::unordered_map<std::string, std::shared_ptr<Port> > ports;
    std::unordered_map<std::string, std::shared_ptr<Module> > submodules; 
    std::shared_ptr<Grid> subGrid;
    std::unordered_map<std::string, int> subModuleNum;
    std::vector<std::string> modeNames;

    std::vector<std::shared_ptr<Module>> global_sw;

};



} // namespace database

#endif //MODULE_H