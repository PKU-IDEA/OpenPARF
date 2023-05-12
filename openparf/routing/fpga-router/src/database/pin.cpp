#include "pin.h"

namespace database {

Pin::Pin(std::string _name, Port* port) {
    name = _name;
    pinCost = 0;
    GSWConnectlength = 0;
    pinPort = port;
}

// void Pin::addConnect(std::shared_ptr<Connect> connect) {
//     connects.push_back(connect);
//     return;
// } 

void Pin::addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType) {
    // std::shared_ptr<Connect> connect(new Connect(connectName, connectOut, connectType));
    connects.push_back(connectOut);
    connectlength.push_back(0);
    connectDelay.push_back(0);
    return;
}

void Pin::addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType, COST_T length) {
    // std::shared_ptr<Connect> connect(new Connect(connectName, connectOut, connectType));
    connects.push_back(connectOut);
    connectlength.push_back(length);
    connectDelay.push_back(length);
    return;
}

void Pin::addPinCost() {
    pinCost += 1;
}


} // namespace database