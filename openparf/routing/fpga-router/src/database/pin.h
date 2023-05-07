#ifndef PIN_H
#define PIN_H
#include <vector>
#include <string>
#include <memory>

#include "connect.h"
#include "utils/utils.h"

namespace database {

class Port;

class Pin {
public:
    Pin(){}
    ~Pin(){
        // std::cout << "Deleting Pin " + name << std::endl;
        // connects.clear();
    }
    Pin(std::string _name, Port* port);
    Pin(std::string _name, INDEX_T x, INDEX_T y, Port* port) : name(_name), pos(x, y), pinCost(0), pinPort(port) {}
    std::string getName() {return name;}
    int getConnectSize() { return connects.size(); }
    std::shared_ptr<Pin> getConnectPinByIdx(int id) {return connects[id];}
    COST_T& connectDelayByIdx(int id) { return connectDelay[id]; }
    COST_T  getConnectDelayByIdx(int id) { return connectDelay[id]; }
    // std::shared_ptr<Connect> getConnectByIdx(int id) {return connects[id];}
    COST_T getPinCost() { return pinCost; }

    // void addConnect(std::shared_ptr<Connect> connect);
    void addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType);
    void addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType, COST_T length);
    void addPinCost();
    void setGSWConnectPin(std::string name) { GSWConnectPin = name; }
    std::string getGSWConnectPin() { return GSWConnectPin; }
    void setGSWConnectDirection(std::string direction) { GSWConnectDirection = direction; }
    std::string getGSWConnectDirection() { return GSWConnectDirection; }
    void setGSWConnectLength(int length) { GSWConnectlength = length; }
    int getGSWConnectLength() { return GSWConnectlength; }
    void setGSWConnectDelay(COST_T delay) { GSWConnectDelay = delay; }
    COST_T getGSWConnectDelay() { return GSWConnectDelay; }

    XY<INDEX_T> getPos() { return pos; }
    void setPinId(int id) { pinId = id; }
    int getPinId() { return pinId; }

    Port* getPinPort() { return pinPort; }

private:
    std::vector<std::shared_ptr<Pin> > connects;
    std::vector<COST_T> connectDelay;
    std::vector<COST_T> connectlength;
    std::string name;
    COST_T pinCost;
    XY<INDEX_T> pos;
    std::string GSWConnectPin;
    std::string GSWConnectDirection;
    int GSWConnectlength;
    COST_T GSWConnectDelay;
    int pinId;

    Port* pinPort;

};
} //namespace database

#endif //PIN_H