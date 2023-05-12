#ifndef CONNECT_H
#define CONNECT_H
#include <string>
#include <memory>

#include <iostream>
#include "utils/utils.h"
namespace database {

enum ConnectType {
    DIRECT,
    CONNNECT,
    BROADCAST
};

class Pin;

class Connect {
public:
    Connect(){}
    Connect(std::string _name, std::shared_ptr<Pin> _output, ConnectType _type);
    Connect(std::string _name, std::shared_ptr<Pin> _output, ConnectType _type, COST_T wl);

    COST_T getCost() { return cost; }
    std::shared_ptr<Pin> getOutput() {return output;}
    void updateCost() {cost *= 2;}
    ~Connect(){
        // std::cout << "Deleting Connect " + name;
        // output = nullptr;
    }

private:
    ConnectType type;
    std::shared_ptr<Pin> output;
    std::string name;
    COST_T wireLength;
    COST_T cost;
};

} //namespace database

#endif //CONNECT_H