#include "port.h"

namespace database {

Port::Port(std::string _name, PortType _portType, int _width = 1, std::string _type = "") {
    name = _name;
    portType = _portType;
    width = _width;
    type = _type;
    pins.resize(0);
    // allPorts[name] = this;
    for (int i = 0; i < width; i++) {
        pins.push_back(std::shared_ptr<Pin>(new Pin(name + "[" + std::to_string(i) + "]", this)));
    }
    // std::cout << _name << ' ' << pins.size() << std::endl;
}

Port::Port(std::string _name, PortType _portType, int _width, std::string _type, INDEX_T x, INDEX_T y)
           :name(_name), portType(_portType), width(_width), type(_type), pos(x, y) {
    pins.resize(0);
    for (int i = 0; i < width; i++) {
        pins.push_back(std::shared_ptr<Pin>(new Pin(name + "[" + std::to_string(i) + "]", x, y, this)));
    }
    }
} // namespace database