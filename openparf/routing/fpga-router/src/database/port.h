#ifndef PORT_H
#define PORT_H

#include <string>
#include <memory>
#include <unordered_map>
#include <iostream>
#include "pin.h"
namespace database {

enum PortType {
    INPUT,
    OUTPUT,
    WIRE,
    INPIN // In order to differentiate from IPIN of router::VertexType
};

class Port {

public:
    std::string name;
    
    Port() {}
    Port(std::string _name, PortType _portType, int _width, std::string _type);
    Port(std::string _name, PortType _portType, int _width, std::string _type, INDEX_T x, INDEX_T y);
    ~Port(){}

    std::shared_ptr<Pin> getPinByIdx(int idx) {
        if (idx >= width) {
            std::cout << "[Error] Port " << name <<  " Pin Index " <<  idx << "is larger than port Width (" << width << "), exiting...\n";
            exit(1);
        }
        return pins[idx]; 
    }
    int getPinNum() {return pins.size();}
    std::string getName() { return name; }
    int getWidth() {return width;}
    PortType getPortType() { return portType; }
    // static std::unordered_map<std::string, std::shared_ptr<Port>> allPorts;

    int setModeBelong(int mode) { modeBelong = mode; }
    int getModeBelong() {return modeBelong; }

    
    XY<INDEX_T> getPos() { return pos; }
private:
    int width;
    PortType portType;
    std::string type;
    std::vector<std::shared_ptr<Pin> >pins;

    int modeBelong;
    XY<INDEX_T> pos;
};


}

#endif //PORT_H