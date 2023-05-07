#ifndef INST_H_
#define INST_H_

#include "database/pin.h"
#include <vector>
#include "net.h"

namespace router {
class Inst;
class InstList {
public:
    InstList() {}
    ~InstList() {}

    void calcDelayAndSlack();
    std::vector<Inst>& getInsts() { return insts; }
    void printSTA();

    COST_T getTNS() { return tns; }
    COST_T getWNS() { return wns; }

    static COST_T period;

private:
    std::vector<Inst> insts;
    COST_T wns = 0;
    COST_T tns = 0;

};

class Inst {
public:
    Inst(){}
    ~Inst(){}

    Inst(std::string _name, INDEX_T id) : name(_name), instId(id), inputNetCnt(0), outputNetCnt(0) {};
    std::string getName() { return name; }
    std::vector<int>& getInputPins() { return inputPins; }
    std::vector<int>& getOutputPins() { return outputPins; }
    int getClockPin() { return clockPin; }
    void setClockPin(int pinId) { clockPin = pinId; }
    std::vector<std::pair<std::pair<int, int>, COST_T> >& getDelayEdges() { return delayEdges; }

    void calcSlack();
    // std::vector<std::shared_ptr<Net> >& getInputNets() { return inputNets; }
    // std::vector<std::shared_ptr<Net> >& getOutputNets() { return outputNets; }
    void addInputNetNum() { inputNetCnt++; }
    void addOutputNetNum() { outputNetCnt++; }

    void addInputPin(int pinId) {
        inputPins.push_back(pinId);
        inputsAT.push_back(0);
        inputsRAT.push_back(std::numeric_limits<COST_T>::max());
        inputsSlack.push_back(0);
    }

    
    void addOutputPin(int pinId) {
        outputPins.push_back(pinId);
        outputsAT.push_back(0);
        outputsRAT.push_back(std::numeric_limits<COST_T>::max());
        outputsSlack.push_back(0);
    }

    bool setTerminal(bool v) { terminal = v; }

    friend class InstList;

private:
    std::string name;
    INDEX_T instId;
    std::vector<int> inputPins;
    std::vector<int> outputPins;
    std::vector<COST_T> inputsAT;
    std::vector<COST_T> outputsAT;
    std::vector<COST_T> inputsRAT;
    std::vector<COST_T> outputsRAT;
    int clockPin;
    std::vector<COST_T> inputsSlack;
    std::vector<COST_T> outputsSlack;
    std::vector<std::pair<std::pair<int, int>, COST_T> > delayEdges; 
    
    bool terminal;
    int inputNetCnt;
    int outputNetCnt;
};
    
} // namespace router


#endif