#ifndef PIN_H
#define PIN_H
#include <vector>
#include <string>
#include <memory>

#include "utils/utils.h"

namespace database {
class Pin {
public:
    Pin(){}
    ~Pin(){
        // std::cout << "Deleting Pin " + name << std::endl;
        // connects.clear();
    }
    // Pin(std::string _name, Port* port);
    Pin(std::string _name, INDEX_T x, INDEX_T y, void* port) : name(_name), pos(x, y), pinCost(0) {}
    std::string getName() {return name;}
    int getConnectSize() { return connects.size(); }
    std::shared_ptr<Pin> getConnectPinByIdx(int id) {return connects[id];}
    // std::shared_ptr<Connect> getConnectByIdx(int id) {return connects[id];}
    COST_T getPinCost() { return pinCost; }

    // void addConnect(std::shared_ptr<Connect> connect);
    // void addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType);
    // void addConnect(std::string connectName, std::shared_ptr<Pin> connectOut, ConnectType connectType, COST_T length);
    void addPinCost();
    XY<INDEX_T> getPos() { return pos; }
    void setPinId(int id) { pinId = id; }
    int getPinId() { return pinId; }

    // Port* getPinPort() { return pinPort; }

private:
    std::vector<std::shared_ptr<Pin> > connects;
    std::vector<COST_T> connectlength;
    std::string name;
    COST_T pinCost;
    XY<INDEX_T> pos;
    int pinId;

    // Port* pinPort;
};
} //namespace database

#endif //PIN_H