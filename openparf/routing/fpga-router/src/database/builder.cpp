#include "builder.h"
#include "port.h"
#include <assert.h>
#include <vector>

#include <iostream>
namespace database {

// std::unordered_map<std::string, std::shared_ptr<Port>> Port::allPorts;
std::unordered_map<std::string, int> Module::GSWInterConnectLength;

std::shared_ptr<Module> buildChip(pugi::xml_node archInfo) {
    pugi::xml_node chipInfo = archInfo.child("chip");
    std::string chipName = chipInfo.attribute("name").value();
    // std::cout << "demo_chip" << std::endl;
    std::shared_ptr<Module> chip(new Module(chipName));
    for (pugi::xml_node input : chipInfo.children("input")) {
        chip->addPort(input.attribute("name").value(), INPUT, input.attribute("width").as_int(), "");
    }
    for (pugi::xml_node output : chipInfo.children("output")) {
        chip->addPort(output.attribute("name").value(), OUTPUT, output.attribute("width").as_int(), "");
    }
    
    int gridWidth = chipInfo.child("grid").attribute("width").as_int();
    int gridHeight = chipInfo.child("grid").attribute("height").as_int();
    chip->setSubGrid(chipInfo.child("grid").attribute("name").value(), gridWidth, gridHeight);
    pugi::xml_node chipGridInfo = chipInfo.child("grid");

    for (pugi::xml_node inst : chipGridInfo.children("inst")) {
        int x = inst.attribute("x").as_int();
        int y = inst.attribute("y").as_int();
        std::string coreType = inst.attribute("type").value();
        std::string coreName = inst.attribute("name").value();
        std::shared_ptr<Module> core = buildCore(coreType, chipName + "." + coreName, archInfo);
        chip->getSubGrid()->setGridModule(core, coreName, x, y, 1, 1);
        chip->addSubModule(coreName, core);
    }

    pugi::xml_node inter = chipInfo.child("interconnect");
    addInterconnect(inter, chip);
    return chip;
}

std::shared_ptr<Module> buildCore(std::string typeName, std::string coreName, pugi::xml_node archInfo) {
    for(pugi::xml_node coreInfo : archInfo.children("core")) {
        std::string name = coreInfo.attribute("name").value();
        if (name == typeName) {
            std::shared_ptr<Module> core(new Module(coreName));
            for (pugi::xml_node input : coreInfo.children("input")) {
                core->addPort(input.attribute("name").value(), INPUT, input.attribute("width").as_int(), "");
            }
            for (pugi::xml_node output : coreInfo.children("output")) {
                core->addPort(output.attribute("name").value(), OUTPUT, output.attribute("width").as_int(), "");
            }

            for (pugi::xml_node gridInfo : coreInfo.children("grid")) {
                std::string type = gridInfo.attribute("type").value();
                //std::cout << type << std::endl;
                if (type == "TILE") {
                    //std::cout << "DD" << std::endl;
                    buildCoreGrid(core, coreName, gridInfo, archInfo);
                }
            }

            pugi::xml_node inter = coreInfo.child("interconnect");
            addInterconnect(inter, core);

            return core;
        }
    }

    return nullptr;
}

void buildCoreGrid(std::shared_ptr<Module> core, std::string coreName, pugi::xml_node gridInfo, pugi::xml_node archInfo) {
    int width = gridInfo.attribute("width").as_int();
    int height = gridInfo.attribute("height").as_int();
    std::string name = gridInfo.attribute("name").value();

    core->setSubGrid(name, width, height);
    
    pugi::xml_node defaultInfo = gridInfo.child("default");
    int dPri = defaultInfo.attribute("pri").as_int();
    core->getSubGrid()->setPriority(dPri, 0, width - 1, 0, height - 1);
    for (pugi::xml_node regionInfo : gridInfo.children("region")) {
        core->getSubGrid()->setPriority(regionInfo.attribute("pri").as_int(), regionInfo.attribute("start_x").as_int(),
                                        regionInfo.attribute("end_x").as_int(), regionInfo.attribute("start_y").as_int(),
                                        regionInfo.attribute("end_y").as_int());
    } 
    for (pugi::xml_node instInfo : gridInfo.children("inst")) {
        core->getSubGrid()->setPriority(instInfo.attribute("pri").as_int(), instInfo.attribute("start_x").as_int(),
                                        instInfo.attribute("start_x").as_int(), instInfo.attribute("end_x").as_int(),
                                        instInfo.attribute("end_x").as_int());
    }
    

    pugi::xml_node dTileInfo = getTile(defaultInfo.attribute("type").value(), archInfo);
    int dWidth = dTileInfo.attribute("width").as_int();
    int dHeight = dTileInfo.attribute("height").as_int();
    for (int i = 0; i < width; i += dWidth)
        for (int j = 0; j < height; j += dHeight) {
            if(core->getSubGrid()->getPriority(i, i + dWidth - 1, j, j + dHeight - 1) == dPri) {
                std::shared_ptr<Module> dModule = buildModule(coreName + ".default[" + std::to_string(i) + "][" + std::to_string(j) + "]", dTileInfo, archInfo, core->getSubGrid(), i, j, dWidth, dHeight);
                core->addSubModule("default[" + std::to_string(i) + "][" + std::to_string(j) + "]", dModule);
                core->getSubGrid()->setGridModule(dModule, "default", i, j, dWidth, dHeight);
            }
        }

    for (pugi::xml_node regionInfo : gridInfo.children("region")) {
        //std::cout << "dd" << std::endl;
        int pri = regionInfo.attribute("pri").as_int();
        int start_x = regionInfo.attribute("start_x").as_int();
        int end_x = regionInfo.attribute("end_x").as_int();
        int start_y = regionInfo.attribute("start_y").as_int();
        int end_y = regionInfo.attribute("end_y").as_int();

        std::string tileType = regionInfo.attribute("type").value();
        std::string tileName = regionInfo.attribute("name").value();
        // std::cout << tileName << ' ' << tileType << ' ' << pri << ' ' << start_x << ' ' << end_x << ' ' << start_y <<' ' << end_y << std::endl;
        pugi::xml_node tileInfo = getTile(tileType, archInfo);
        int tileWidth = tileInfo.attribute("width").as_int();
        int tileHeight = tileInfo.attribute("height").as_int();

        for (int i = start_x; i <= end_x; i += tileWidth)
            for (int j = start_y; j <= end_y; j += tileHeight) {
                // std::cout << i << ' ' << j << ' ' << core->getSubGrid()->getPriority(i, i + tileWidth - 1, j, j + tileHeight - 1) << std::endl;
                if (core->getSubGrid()->getPriority(i, i + tileWidth - 1, j, j + tileHeight - 1) == pri) {
                    std::shared_ptr<Module> tile = buildModule(coreName + "." + tileName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]", tileInfo, archInfo, core->getSubGrid(), i, j, tileWidth, tileHeight);
                    // std::cout << i << ' ' << j << ' ' << tileWidth << ' ' << tileHeight << std::endl;
                    core->addSubModule(tileName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]", tile);
                    core->getSubGrid()->setGridModule(tile, tileName, i, j, tileWidth, tileHeight);
                }
            }
    }

     for (pugi::xml_node instInfo : gridInfo.children("inst")) {
        int pri = instInfo.attribute("pri").as_int();
        int start_x = instInfo.attribute("start_x").as_int();
        int end_x = instInfo.attribute("end_x").as_int();

        std::string tileType = instInfo.attribute("type").value();
        pugi::xml_node tileInfo = getTile(tileType, archInfo);
        int tileWidth = tileInfo.attribute("width").as_int();
        int tileHeight = tileInfo.attribute("height").as_int();

        if (core->getSubGrid()->getPriority(start_x, start_x + tileWidth - 1, end_x, end_x + tileHeight - 1) == pri) {
            std::shared_ptr<Module> tile = buildModule(coreName + "." + "DUMMY" + "[" + std::to_string(start_x) + "][" + std::to_string(end_x) + "]", tileInfo, archInfo, core->getSubGrid(), start_x, end_x, tileWidth, tileHeight);
            core->addSubModule("DUMMY[" + std::to_string(start_x) + "][" + std::to_string(end_x) + "]", tile);
            core->getSubGrid()->setGridModule(tile, "DUMMY", start_x, end_x, tileWidth, tileHeight);
        }
    }

    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            if((i == 0 || i == width - 1) && (j == 0 || j == width - 1)) continue;
            buildGSWInterConnect(core->getSubGrid(), i, j, 1, NORTH);
            buildGSWInterConnect(core->getSubGrid(), i, j, 2, NORTH);
            buildGSWInterConnect(core->getSubGrid(), i, j, 6, NORTH);
            
            buildGSWInterConnect(core->getSubGrid(), i, j, 1, SOUTH);
            buildGSWInterConnect(core->getSubGrid(), i, j, 2, SOUTH);
            buildGSWInterConnect(core->getSubGrid(), i, j, 6, SOUTH);
            
            buildGSWInterConnect(core->getSubGrid(), i, j, 1, EAST);
            buildGSWInterConnect(core->getSubGrid(), i, j, 2, EAST);
            buildGSWInterConnect(core->getSubGrid(), i, j, 6, EAST);
            
            buildGSWInterConnect(core->getSubGrid(), i, j, 1, WEST);
            buildGSWInterConnect(core->getSubGrid(), i, j, 2, WEST);
            buildGSWInterConnect(core->getSubGrid(), i, j, 6, WEST);
        }

}

std::shared_ptr<Module> buildModule(std::string moduleName, pugi::xml_node moduleInfo, pugi::xml_node archInfo, std::shared_ptr<Grid> grid, int x, int y, int width, int height) {
    std::shared_ptr<Module> module(new Module(moduleName));
    for (pugi::xml_node input : moduleInfo.children("input")) {
        std::string portType = input.attribute("type").value();
        // std::cout << "portType: " << portType << std::endl; 
        if (portType == "IPIN")
            module->addPort(input.attribute("name").value(), INPIN, input.attribute("width").as_int(), "", x, y);
        else 
            module->addPort(input.attribute("name").value(), INPUT, input.attribute("width").as_int(), "", x, y);
        
        if (input.attribute("output")) {
            module->addPort(input.attribute("output").value(), OUTPUT, input.attribute("width").as_int(), "", x, y);
        }
    }
    for (pugi::xml_node output : moduleInfo.children("output")) {
        std::string portType = output.attribute("type").value();
        if (portType == "OPIN")
            module->addPort(output.attribute("name").value(), INPIN, output.attribute("width").as_int(), "", x, y);
        else 
            module->addPort(output.attribute("name").value(), OUTPUT, output.attribute("width").as_int(), "", x, y);
    }
    for (pugi::xml_node wire : moduleInfo.children("wire")) {
        module->addPort(wire.attribute("name").value(), WIRE, std::max(1, wire.attribute("width").as_int()), "", x, y);
    }

    for (pugi::xml_node subModInfo : moduleInfo.children("module")) {
        std::string subName = subModInfo.attribute("name").value();
        int num = subModInfo.attribute("num").as_int();
        module->setSubModuleNum(subName, num);
        for (int i = 0; i < num; i++) {
            std::shared_ptr<Module> sub = buildModule(moduleName + "." + subName + "[" + std::to_string(i) + "]" ,subModInfo, archInfo, grid, x, y, width, height);
            module->addSubModule(subName + "[" + std::to_string(i) + "]", sub);
        }
    }

    for (pugi::xml_node subModInfo : moduleInfo.children("local_sw")) {
        std::string subName = subModInfo.attribute("name").value();
        int num = subModInfo.attribute("num").as_int();
        module->setSubModuleNum(subName, num);
        for (int i = 0; i < num; i++) {
            std::shared_ptr<Module> sub = buildModule(moduleName + "." + subName + "[" + std::to_string(i) + "]" ,subModInfo, archInfo, grid, x, y, width, height);
            module->addSubModule(subName + "[" + std::to_string(i) + "]", sub);        }
    }

    int modeID = 0;
    for (pugi::xml_node modeInfo : moduleInfo.children("mode")) {
        std::string modeName = modeInfo.attribute("name").value();
        module->addMode(modeName);
        for (pugi::xml_node instInfo : modeInfo.children("inst")) {
            std::string instName = instInfo.attribute("name").value();
            std::string instType = instInfo.attribute("type").value();
            int num = instInfo.attribute("num").as_int();
            pugi::xml_node instModuleInfo;
            instModuleInfo = getPrim(instType, archInfo);
            module->setSubModuleNum(instName, num);
            for (int i = 0; i < num; i++) {
                std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(i) + "]" ,instModuleInfo, archInfo, grid, x, y, width, height);
                sub->setModeBelong(modeID);
                module->addSubModule(instName + "[" + std::to_string(i) + "]", sub);   
                assert (module->getSubModule(instName + "[" + std::to_string(i) + "]") != nullptr);
            }
        }
        pugi::xml_node inter = modeInfo.child("interconnect");
        addInterconnect(inter, module);
        modeID++;
    }

    for (pugi::xml_node instInfo : moduleInfo.children("inst")) {
        std::string instName = instInfo.attribute("name").value();
        std::string instType = instInfo.attribute("type").value();
        int num = instInfo.attribute("num").as_int();
        // std::cout << instName << '|' << instType << std::endl;
        pugi::xml_node instModuleInfo;
        if (instName == "gsw") {
            instModuleInfo = getGSW(instType, archInfo);
                module->setSubModuleNum(instName, num);
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++) {
                    int id = i * width + j;
                    std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(id) + "]" ,instModuleInfo, archInfo);
                    grid->setGridGSW(x + i, y + j, sub);
                    // std::cout << "x = " << x << " y = " << y << " i = " << i << " j = " << j << " name :" << sub->getName() << " " << grid->getGridGSW(x + i, y + j) -> getName() << std::endl;
                    module->addSubModule(instName + "[" + std::to_string(id) + "]", sub);   
                    assert (module->getSubModule(instName + "[" + std::to_string(id) + "]") != nullptr);
                }
        } else {
                instModuleInfo = getPrim(instType, archInfo);
                module->setSubModuleNum(instName, num);
                for (int i = 0; i < num; i++) {
                    std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(i) + "]" ,instModuleInfo, archInfo);
                    module->addSubModule(instName + "[" + std::to_string(i) + "]", sub);   
                    assert (module->getSubModule(instName + "[" + std::to_string(i) + "]") != nullptr);
                }
        }
    }

    pugi::xml_node inter = moduleInfo.child("interconnect");
    addInterconnect(inter, module);

    return module;
}

std::shared_ptr<Module> buildModule(std::string moduleName, pugi::xml_node moduleInfo, pugi::xml_node archInfo) {
    //  std::cout << "BuildModule                    " << moduleName << std::endl;
    std::shared_ptr<Module> module(new Module(moduleName));
    // std::cout << "SS" << std::endl;
    for (pugi::xml_node input : moduleInfo.children("input")) {
        std::shared_ptr<Port> port;
        std::string portType = input.attribute("type").value();
        if (portType == "IPIN")
            port = module->addPort(input.attribute("name").value(), INPIN, input.attribute("width").as_int(), "");
        else 
            port = module->addPort(input.attribute("name").value(), INPUT, input.attribute("width").as_int(), "");
        if (input.attribute("output")) {
            std::string outputName = input.attribute("output").value();
            // if (outputName == "o_lsw_special") {
            //     std::cout << input.attribute("direction").value() << ' ' << input.attribute("length").as_int() << std::endl;
            //     getchar();
            // }
            int width = input.attribute("width").as_int();
            module->addPort(outputName, OUTPUT, width, "");
            for (int i = 0; i < width; i++) {
                port->getPinByIdx(i)->setGSWConnectPin(outputName + "[" + std::to_string(i) + "]");
                port->getPinByIdx(i)->setGSWConnectDirection(input.attribute("direction").value());
                port->getPinByIdx(i)->setGSWConnectLength(input.attribute("length").as_int());
                int length = input.attribute("length").as_int();
                if (length == 1) port->getPinByIdx(i)->setGSWConnectDelay(50);
                if (length == 2) port->getPinByIdx(i)->setGSWConnectDelay(60);
                if (length == 6) port->getPinByIdx(i)->setGSWConnectDelay(120);
            }
        }
    }
    for (pugi::xml_node output : moduleInfo.children("output")) {
        std::string portType = output.attribute("type").value();
        if (portType == "OPIN")
            module->addPort(output.attribute("name").value(), INPIN, output.attribute("width").as_int(), "");
        else 
            module->addPort(output.attribute("name").value(), OUTPUT, output.attribute("width").as_int(), "");
    }
    for (pugi::xml_node wire : moduleInfo.children("wire")) {
        module->addPort(wire.attribute("name").value(), WIRE, std::max(1, wire.attribute("width").as_int()), "");
    }

    for (pugi::xml_node subModInfo : moduleInfo.children("module")) {
        std::string subName = subModInfo.attribute("name").value();
        int num = subModInfo.attribute("num").as_int();
        module->setSubModuleNum(subName, num);
        for (int i = 0; i < num; i++) {
            std::shared_ptr<Module> sub = buildModule(moduleName + "." + subName + "[" + std::to_string(i) + "]" ,subModInfo, archInfo);
            module->addSubModule(subName + "[" + std::to_string(i) + "]", sub);
        }
    }

    for (pugi::xml_node subModInfo : moduleInfo.children("local_sw")) {
        std::string subName = subModInfo.attribute("name").value();
        int num = subModInfo.attribute("num").as_int();
        module->setSubModuleNum(subName, num);
        for (int i = 0; i < num; i++) {
            std::shared_ptr<Module> sub = buildModule(moduleName + "." + subName + "[" + std::to_string(i) + "]" ,subModInfo, archInfo);
            module->addSubModule(subName + "[" + std::to_string(i) + "]", sub);        }
    }

    int modeID = 0;
    for (pugi::xml_node modeInfo : moduleInfo.children("mode")) {
        std::string modeName = modeInfo.attribute("name").value();
        module->addMode(modeName);
        for (pugi::xml_node instInfo : modeInfo.children("inst")) {
            std::string instName = instInfo.attribute("name").value();
            std::string instType = instInfo.attribute("type").value();
            int num = instInfo.attribute("num").as_int();
            // std::cout << instName << '|' << instType << std::endl;
            pugi::xml_node instModuleInfo;
            instModuleInfo = getPrim(instType, archInfo);
            module->setSubModuleNum(instName, num);
            for (int i = 0; i < num; i++) {
                std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(i) + "]" ,instModuleInfo, archInfo);
                sub->setModeBelong(modeID);
                module->addSubModule(instName + "[" + std::to_string(i) + "]", sub);   
                assert (module->getSubModule(instName + "[" + std::to_string(i) + "]") != nullptr);
            }
        }
        pugi::xml_node inter = modeInfo.child("interconnect");
        addInterconnect(inter, module);
        modeID++;
    }

    for (pugi::xml_node instInfo : moduleInfo.children("inst")) {
        std::string instName = instInfo.attribute("name").value();
        std::string instType = instInfo.attribute("type").value();
        int num = instInfo.attribute("num").as_int();
        // std::cout << instName << '|' << instType << std::endl;
        pugi::xml_node instModuleInfo;
        if (instName == "gsw") {
            instModuleInfo = getGSW(instType, archInfo);
            module->setSubModuleNum(instName, num);
            for (int i = 0; i < num; i++) {
                std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(i) + "]" ,instModuleInfo, archInfo);
                module->addSubModule(instName + "[" + std::to_string(i) + "]", sub);   
                assert (module->getSubModule(instName + "[" + std::to_string(i) + "]") != nullptr);
                module->addGSW(sub);
            }
        } else {
            instModuleInfo = getPrim(instType, archInfo);
            module->setSubModuleNum(instName, num);
            for (int i = 0; i < num; i++) {
                std::shared_ptr<Module> sub = buildModule(moduleName + "." + instName + "[" + std::to_string(i) + "]" ,instModuleInfo, archInfo);
                module->addSubModule(instName + "[" + std::to_string(i) + "]", sub);   
                assert (module->getSubModule(instName + "[" + std::to_string(i) + "]") != nullptr);
            }
        }
    }

    pugi::xml_node inter = moduleInfo.child("interconnect");
    addInterconnect(inter, module);

    return module;
}

pugi::xml_node getTile(std::string tileType, pugi::xml_node archInfo) {
    for (pugi::xml_node tileInfo : archInfo.child("tile_blocks").children("tile")) {
        if (tileInfo.attribute("type").value() == tileType)
            return tileInfo;
    }
    assert(0);
    return archInfo;
}


pugi::xml_node getPrim(std::string PrimType, pugi::xml_node archInfo) {
    // std::cout << PrimType << std::endl;
    for (pugi::xml_node primInfo : archInfo.child("primitives").children("primitive")) {
        // std::cout << primInfo.attribute("name").value() << std::endl;
        if (primInfo.attribute("name").value() == PrimType)
            return primInfo;
    }
    // std::cerr << PrimType << std::endl;
    std::cerr << "[error] PrimType "  << PrimType << " Not Found!, exiting..." << std::endl;
    assert(0);
    // exit(0);
    return archInfo;
}

pugi::xml_node getGSW(std::string GSWType, pugi::xml_node archInfo) {
    for (pugi::xml_node GSWInfo : archInfo.children("global_sw")) {
        if (GSWInfo.attribute("name").value() == GSWType)
            return GSWInfo;
    }
    assert(0);
    return archInfo;
}

void addInterconnect(pugi::xml_node inter, std::shared_ptr<Module> module) {
    for (pugi::xml_node direct : inter.children("direct")) {
        std::string inputs = direct.attribute("inputs").value();
        std::string outputs = direct.attribute("outputs").value();
        std::string name = direct.attribute("name").value();
        addDirect(inputs, outputs, module, name);
    }
    for (pugi::xml_node direct : inter.children("connect")) {
        std::string inputs = direct.attribute("inputs").value();
        std::string outputs = direct.attribute("outputs").value();
        std::string name = direct.attribute("name").value();
        addConnect(inputs, outputs, module, name);
    }
    for (pugi::xml_node direct : inter.children("broadcast")) {
        std::string inputs = direct.attribute("inputs").value();
        std::string outputs = direct.attribute("outputs").value();
        std::string name = direct.attribute("name").value();
        addBroadcast(inputs, outputs, module, name);
    }
}

void addDirect(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName) {
    std::string currentModuleName = currentModule->getName();
    // std::cout << inputs << "|" << outputs << std::endl;
    std::vector<std::shared_ptr<Pin> > inputPins;
    getPins(inputs, currentModule, inputPins);
    std::vector<std::shared_ptr<Pin> > outputPins;
    getPins(outputs, currentModule, outputPins);
    //std::cout << "EEE" << std::endl;
    // if (inputPins.size() != outputPins.size()) 
        // std::cout << inputs << "|" << outputs << ' ' << inputPins.size() << ' ' << outputPins.size() <<  std::endl;
    if (inputPins.size() != outputPins.size()) {
        std::cout << "[Warning] Input Size not equals to Output Size!" << std::endl;
        std::cout << inputs << ' ' << outputs << std::endl;
        std::cout << inputPins.size() << ' ' << outputPins.size() << std::endl;
        // return;
    }
    // assert(inputPins.size() == outputPins.size());
    //std::cout << "FFF" << std::endl;
    int width = std::min(inputPins.size(), outputPins.size());
    //std::cout << ' ' << width <<std::endl;
    for (int i = 0; i < width; i++) {
       // std::cout << i << ' ' << connectName << ' ' << inputPins[i]->getName() << ' ' << outputPins[i]->getName() << std::endl;
        inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "]", outputPins[i], DIRECT);
    }
}

void addConnect(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName) {
    std::string currentModuleName = currentModule->getName();
    // std::cout << inputs << ' ' << outputs << std::endl;
    std::vector<std::shared_ptr<Pin> > inputPins;
    getPins(inputs, currentModule, inputPins);
    std::vector<std::shared_ptr<Pin> > outputPins;
    getPins(outputs, currentModule, outputPins);

    int widthIn = inputPins.size();
    int widthOut = outputPins.size();
    for (int i = 0; i < widthIn; i++) {
        // std::cout << (inputPins[i]==nullptr) << std::endl;
        for (int j = 0; j < widthOut; j++) {
        if (currentModuleName.find("lsw") != std::string::npos) {
            if (currentModuleName.find("FU") == std::string::npos) {
                if (outputs.find("o_fu_ctrl") != std::string::npos || outputs.find("o_gsw_switch") != std::string::npos || outputs.find("h_fan") != std::string::npos) {
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT, 40);
                }
                else if (outputs.find("o_fu") != std::string::npos)
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT, 50);
                else
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT);
            }
            else {
                if (outputs.find("o_fu_ctrl") != std::string::npos  || outputs.find("h_fan") != std::string::npos) {
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT, 40);
                }
                else if (outputs.find("o_fu") != std::string::npos || outputs.find("o_gsw_switch") != std::string::npos)
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT, 50);
                else
                    inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT);

            }
        }
        else 
        inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], CONNNECT);
        
        }
    }
}

void addBroadcast(std::string inputs, std::string outputs, std::shared_ptr<Module> currentModule, std::string connectName) {
    std::string currentModuleName = currentModule->getName();

    std::vector<std::shared_ptr<Pin> > inputPins;
    getPins(inputs, currentModule, inputPins);
    std::vector<std::shared_ptr<Pin> > outputPins;
    getPins(outputs, currentModule, outputPins);

    int widthIn = inputPins.size();
    int widthOut = outputPins.size();
    for (int i = 0; i < widthIn; i++) {
        for (int j = 0; j < widthOut; j++) {
        inputPins[i]->addConnect(connectName + "[" + std::to_string(i) + "][" + std::to_string(j) + "]" , outputPins[j], BROADCAST);
        }
    }
}

void getPins(std::string ports, std::shared_ptr<Module> currentModule, std::vector<std::shared_ptr<Pin>>& ret) {
    std::string remainPorts = ports;
    std::string now;
    
    bool isEnd = false;
    while (!isEnd) {
        int pos = remainPorts.find(",");
        if (pos != std::string::npos) {
            now = remainPorts.substr(0, pos);
            remainPorts = remainPorts.substr(pos + 1);
        }
        else {
            now = remainPorts;
            isEnd = true;
        }
        getPinsFromPort(now, currentModule, ret);
    }
}

void getPinsFromPort(std::string port, std::shared_ptr<Module> currentModule, std::vector<std::shared_ptr<Pin>>& ret) {
    
    if (port.find(".") == std::string::npos) {
        int strLen = port.size();
        if(port[port.size() - 1] == ']') {
            if (port.find(":") != std::string::npos) {
                int pos1 = port.find("[");
                int pos2 = port.find(":");
                int en = std::stoi(port.substr(pos1 + 1, pos2 - pos1 - 1));
                int st = std::stoi(port.substr(pos2 + 1, strLen - pos2 - 2));
                std::string portName = port.substr(0, pos1);
                std::shared_ptr<Port> targetPort = currentModule->getPort(portName);
                if (targetPort == nullptr) {
                    std::cout << "[Warning 1] Port " << portName << " Not Find!" << std::endl;
                    std::cout << port << ' ' << currentModule->getName() << std::endl;
                    currentModule->listSubModules();
                    return;
                }
                for (int i = en; i >= st; i--) {
                    ret.push_back(targetPort->getPinByIdx(i));
                }
            }
            else {
                int pos = port.find("[");
                int idx = std::stoi(port.substr(pos + 1, strLen - pos - 2));
                std::string portName = port.substr(0, pos);
                std::shared_ptr<Port> targetPort = currentModule->getPort(portName);
                if (targetPort == nullptr) {
                    std::cout << "[Warning 2] Port " << portName << " Not Find!" << std::endl;
                    std::cout << port << ' ' << currentModule->getName() << std::endl;
                    return;
                }
                ret.push_back(targetPort->getPinByIdx(idx));
            }
        }
        else {
            std::shared_ptr<Port> targetPort = currentModule->getPort(port);
            if (targetPort == nullptr) {
                std::cout << "[Warning 3] Port " << port << " Not Find!" << std::endl;
                std::cout << port << ' ' << currentModule->getName() << std::endl;
                return;
            }
            int portWidth = targetPort->getWidth();
            for (int i = portWidth - 1; i >= 0; i--) {
                ret.push_back(targetPort->getPinByIdx(i));
            }
        }
    }
    else {
        std::string subModuleName = port.substr(0, port.find("."));
        std::string remainPart = port.substr(port.find(".") + 1);
        if (currentModule->getSubModule(subModuleName) != nullptr) {
            // std::cout << subModuleName << std::endl;
            getPinsFromPort(remainPart, currentModule->getSubModule(subModuleName), ret);
        } else if (currentModule->getSubGrid() != nullptr) {
            // std::cout << currentModule->getName() << std::endl;
            std::shared_ptr<Grid> grid = currentModule->getSubGrid();
            int gridWidth = grid->getWidth();
            int gridHeight = grid->getHeight();
            for (int i = gridWidth - 1; i >= 0; i--) 
                for (int j = gridHeight - 1; j >= 0; j--) {
                    if (grid->getModuleName(i, j) == subModuleName)
                        getPinsFromPort(remainPart, grid->getModule(i, j), ret);
                }
        } else {
            if (subModuleName.find("[") == std::string::npos) {
                int num = currentModule->getSubModuleNum(subModuleName);
                for (int i = num - 1; i >= 0; i--) {
                    getPinsFromPort(remainPart, currentModule->getSubModule(subModuleName + "[" + std::to_string(i) + "]"), ret);
                }
            }else if (subModuleName.find(":") == std::string::npos) {
                std::cout << "[ERROR] SubModule " << subModuleName << " Not Found! Exiting..." << std::endl;
                exit(1);
            } else {
                int pos1 = subModuleName.find("[");
                int pos2 = subModuleName.find(":");
                int pos3 = subModuleName.find("]");
                int en = std::stoi(subModuleName.substr(pos1 + 1, pos2 - pos1 - 1));
                int st = std::stoi(subModuleName.substr(pos2 + 1, pos3 - pos2 - 1));
                std::string subName = subModuleName.substr(0, pos1);
                for (int i = en; i >= st; i--) {
                    getPinsFromPort(remainPart, currentModule->getSubModule(subName + "[" + std::to_string(i) + "]"), ret);
                }
            }
            
        }
    }
}

void buildGSWInterConnect(std::shared_ptr<Grid> grid, int x, int y, int length, GSWInterConnectDirection direction) {
    int width = grid->getWidth(); 
    int height = grid->getHeight();
    std::shared_ptr<Module> currentGSW = grid->getGridGSW(x, y);
    // std::cout << (currentGSW == nullptr) << std::endl;
    // std::cout << x << ' ' << y << ' ' << length << ' ' << direction << ' ' << GSWName << std::endl; 
    if (direction == NORTH) {
        std::shared_ptr<Port> O_North = currentGSW->getPort("o_nl_" + std::to_string(length));
        int headY = (x == 0 || x == width - 1 ? height - 2 : height - 1);
        if (y + length <= headY) {
            std::shared_ptr<Port> I_South = grid->getGridGSW(x, y + length)->getPort("i_sl_" + std::to_string(length));
            for (int i = 0; i < O_North->getWidth(); i++)
                O_North->getPinByIdx(i)->addConnect("InterGSWConnect_N" + std::to_string(i), I_South->getPinByIdx(i), DIRECT, length);
        } else {
            int targetY = headY - (y + length - headY);
            std::shared_ptr<Port> I_South = grid->getGridGSW(x, targetY)->getPort("i_nl_" + std::to_string(length));
            for (int i = 0; i < O_North->getWidth(); i++)
                O_North->getPinByIdx(i)->addConnect("InterGSWConnect_N" + std::to_string(i), I_South->getPinByIdx(i), DIRECT), length;
        }
    } 
    if (direction == SOUTH) {
        std::shared_ptr<Port> O_South = currentGSW->getPort("o_sl_" + std::to_string(length));
        int headY = (x == 0 || x == width - 1 ? 1 : 0);
        if (y - length >= headY) {
            std::shared_ptr<Port> I_North = grid->getGridGSW(x, y - length)->getPort("i_nl_" + std::to_string(length));
            for (int i = 0; i < O_South->getWidth(); i++)
                O_South->getPinByIdx(i)->addConnect("InterGSWConnect_S" + std::to_string(i), I_North->getPinByIdx(i), DIRECT, length);
        } else {
            int targetY = headY + (headY + length - y);
            std::shared_ptr<Port> I_North = grid->getGridGSW(x, targetY)->getPort("i_sl_" + std::to_string(length));
            for (int i = 0; i < O_South->getWidth(); i++)
                O_South->getPinByIdx(i)->addConnect("InterGSWConnect_S" + std::to_string(i), I_North->getPinByIdx(i), DIRECT, length);
        }
    }
    if (direction == EAST) {
        std::shared_ptr<Port> O_East = currentGSW->getPort("o_el_" + std::to_string(length));
        int headX = (y == 0 || y == height - 1 ? width - 2 : width - 1);
        if (x + length <= headX) {
            std::shared_ptr<Port> I_West = grid->getGridGSW(x + length, y)->getPort("i_wl_" + std::to_string(length));
            for (int i = 0; i < O_East->getWidth(); i++)
                O_East->getPinByIdx(i)->addConnect("InterGSWConnect_E" + std::to_string(i), I_West->getPinByIdx(i), DIRECT, length);
        } else {
            int targetX = headX - (x + length - headX);
            std::shared_ptr<Port> I_West = grid->getGridGSW(targetX, y)->getPort("i_el_" + std::to_string(length));
            for (int i = 0; i < O_East->getWidth(); i++)
                O_East->getPinByIdx(i)->addConnect("InterGSWConnect_E" + std::to_string(i), I_West->getPinByIdx(i), DIRECT, length);
        }
    } 
    if (direction == WEST) {
        std::shared_ptr<Port> O_West = currentGSW->getPort("o_wl_" + std::to_string(length));
        int headX = (y == 0 || y == height - 1 ? 1 : 0);
        if (x - length >= headX) {
            std::shared_ptr<Port> I_East = grid->getGridGSW(x - length, y)->getPort("i_el_" + std::to_string(length));
            for (int i = 0; i < O_West->getWidth(); i++)
                O_West->getPinByIdx(i)->addConnect("InterGSWConnect_W" + std::to_string(i), I_East->getPinByIdx(i), DIRECT, length);
        } else {
            int targetX = headX + (headX + length - x);
            std::shared_ptr<Port> I_East = grid->getGridGSW(targetX, y)->getPort("i_wl_" + std::to_string(length));
            for (int i = 0; i < O_West->getWidth(); i++)
                O_West->getPinByIdx(i)->addConnect("InterGSWConnect_W" + std::to_string(i), I_East->getPinByIdx(i), DIRECT, length);
        }
    }
}

// void getPins(std::string ports, std::shared_ptr<Module> currentModule, std::vector<std::shared_ptr<Pin>>& ret) {
//     // std::cout << ports << ' ' << currentModule->getName() << std::endl;
//     //std::vector<std::shared_ptr<Pin> > ret;=
//     std::string remainPorts = ports;
//     std::string now;

//     bool isEnd = false;
//     while(!isEnd) {
//         int pos = remainPorts.find(",");
//         if (pos != std::string::npos) {
//             now = remainPorts.substr(0, pos);
//             remainPorts = remainPorts.substr(pos + 1);
//         }
//         else {
//             now = remainPorts;
//             isEnd = true;
//         }
//         // std::cout << ' ' << now << std::endl;

//         int nowLen = now.size();
//         if (now[nowLen - 1] == ']') {
//             if (now.find(":") != std::string::npos) {
//                 int pos1 = now.find(":"), pos2 = now.find(":");
//                 for (; now[pos2] != '['; pos2--);
//                 int num1 = std::stoi(now.substr(pos2 + 1, pos1 - pos2 - 1));
//                 int num2 = std::stoi(now.substr(pos1 + 1, nowLen - pos1 - 2));
//                 std::string portName = currentModule->getName() + "." + now.substr(0, pos2);
//                 // std::cout << num1 << ' ' << num2 << ' ' << portName << std::endl;
//                 std::shared_ptr<Port> port = Port::allPorts[portName];
//                 if (port != nullptr)
//                     for (int i = num1; i >= num2; i--) {
//                         ret.push_back(port->getPinByIdx(i));
//                     }
//             }
//             else {
//                 int pos1 = nowLen - 1;
//                 for (; now[pos1] != '['; pos1--);
//                 // std::cout << pos1 << std::endl;
//                 int num = std::stoi(now.substr(pos1 + 1, nowLen - pos1 - 2));
//                 std::string portName = currentModule->getName() + "." + now.substr(0, pos1);
//                 // std::cout << portName << std::endl;
//                 std::shared_ptr<Port> port = Port::allPorts[portName];
//                 // std::cout << num << ' ' << (port == nullptr) << std::endl;
//                 // std::cout <<"    " << (port->getPinByIdx(num) == nullptr) << std::endl;
//                 if (port != nullptr)
//                     ret.push_back(port->getPinByIdx(num));
//             }

//         }
//         else {
//             // std::cout << "else" << std::endl;
//             std::string portName = currentModule->getName() + "." + now;
//             if (Port::allPorts.find(portName) == Port::allPorts.end()) {
//                 // std::cout << "if" << std::endl;
//                 std::shared_ptr<Grid> grid = currentModule->getSubGrid();
//                 int width = grid->getWidth();
//                 int height = grid->getHeight();
//                 int pos = now.find(".");
//                 std::string subModuleName = now.substr(0, pos);
//                 std::string portName = now.substr(pos + 1);
//                 // std::cout << subModuleName << ' ' << portName << std::endl;
//                 for (int x = 0; x < width; x++)
//                     for (int y = 0; y < height; y++) {
//                         // std::cout << x << ' ' << y << ' ' << grid->getModuleName(x, y) << ' ' << subModuleName << std::endl;
//                         if (grid->getModuleName(x, y) == subModuleName) {
//                             std::shared_ptr<Port> port = grid->getModule(x, y)->getPort(portName);
//                             if (port != nullptr)
//                                 for (int i = port->getWidth() - 1; i >= 0; i--) {
//                                     ret.push_back(port->getPinByIdx(i));
//                                 }
//                         }
//                     }
//             }
//             else {
//                 std::shared_ptr<Port> port = Port::allPorts[portName];
//                 if(port != nullptr) {
//                     int portWidth = port->getWidth();
//                     for (int i = portWidth - 1; i >= 0; i--) {
//                         ret.push_back(port->getPinByIdx(i));
//                     }
//                 }
//             }
//         }

//     }

//     // std::cout << ret.size() << std::endl;
//     //return ret;
// }

} //namespace database