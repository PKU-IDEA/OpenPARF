#include "module.h"
#include "port.h"

#include <assert.h>
namespace database {


Module::Module(std::string _name) {
    name = _name;
    modeBelong = -1;
    modeSelect = 0;
}

std::shared_ptr<Port> Module::addPort(std::string portName, PortType portType,  int width, std::string portInfo) {
    // std::cout << "Module::addPort " << name << ' ' << portName << std::endl; 
    std::shared_ptr<Port> port(new Port(name + "." + portName, portType, width, portInfo));
    ports[portName] = port;
    // Port::allPorts[port->getName()] = port;
    return port;
}

std::shared_ptr<Port> Module::addPort(std::string portName, PortType portType, int width, std::string portInfo, int x, int y) {
    // std::cout << "Module::addPort " << name << ' ' << portName << std::endl; 
    std::shared_ptr<Port> port(new Port(name + "." + portName, portType, width, portInfo, x, y));
    ports[portName] = port;
    // Port::allPorts[port->getName()] = port;
    return port;
}
void Module::addSubModule(std::string subModuleName, std::shared_ptr<Module> subModule) {
    submodules[subModuleName] = subModule;
}

void Module::setSubGrid(std::string gridName, int width, int height) {
    subGrid = std::shared_ptr<Grid>(new Grid(gridName, width, height));
}

} // namesapce database