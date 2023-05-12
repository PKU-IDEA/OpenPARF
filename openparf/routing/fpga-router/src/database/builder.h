#ifndef BUILDER_H
#define BUILDER_H

#include "module.h"
#include "connect.h"
#include "pin.h"
#include "port.h"

#include <pugixml/pugixml.hpp>
#include <memory>
#include <string>
namespace database {

std::shared_ptr<Module> buildChip(pugi::xml_node archInfo);
std::shared_ptr<Module> buildCore(std::string typeName, std::string coreName, pugi::xml_node archInfo);
std::shared_ptr<Module> buildModule(std::string moduleName, pugi::xml_node moduleInfo, pugi::xml_node archInfo, std::shared_ptr<Grid> grid, int x, int y, int width, int height);
std::shared_ptr<Module> buildModule(std::string moduleName, pugi::xml_node moduleInfo, pugi::xml_node archInfo);
void buildCoreGrid(std::shared_ptr<Module> core, std::string coreName, pugi::xml_node gridInfo, pugi::xml_node archInfo);

void addInterconnect(pugi::xml_node inter, std::shared_ptr<Module> module);
void addDirect(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName);
void addConnect(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName);
void addBroadcast(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName);

pugi::xml_node getTile(std::string tileType, pugi::xml_node archInfo);
pugi::xml_node getPrim(std::string PrimType, pugi::xml_node archInfo);
pugi::xml_node getGSW(std::string GSWType, pugi::xml_node archInfo);

void getPins(std::string ports, std::shared_ptr<Module> currentModule, std::vector<std::shared_ptr<Pin>>& ret);
void getPinsFromPort(std::string port, std::shared_ptr<Module> currentModule, std::vector<std::shared_ptr<Pin>>& ret);

enum GSWInterConnectDirection {
    NORTH,
    SOUTH,
    EAST,
    WEST
};

void buildGSWInterConnect(std::shared_ptr<Grid> grid, int x, int y, int length, GSWInterConnectDirection direction);

}// namespace database

#endif // BUILDER_H