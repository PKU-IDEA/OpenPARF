#include "inst.h"
#include "net.h"
#include "routegraph.h"
#include "pathfinder.h"
#include <queue>
#include <iostream>
#include <fstream>

namespace router {

    COST_T InstList::period = 0;

    void InstList::calcDelayAndSlack() {
        std::cout << "Start calc delay..." << std::endl;
        std::vector<int> visited(insts.size(), 0);
        std::queue<int> q;
        std::vector<COST_T> instDelay(insts.size(), 0);
        std::vector<COST_T> instRAT(insts.size(), std::numeric_limits<COST_T>::max());
        auto& routetree = Pathfinder::routetree;

        for (auto& inst : insts) {
            if (inst.inputNetCnt == 0 || inst.terminal)
                q.push(inst.instId);
        }
        // std::cout << "calc AT..." << std::endl;

        while (!q.empty()) {
            auto& inst = insts[q.front()];
            q.pop();

            for (int i = 0; i < inst.outputsAT.size(); i++)
                inst.outputsAT[i] = instDelay[inst.instId];
            if (!inst.terminal)
                for (auto& it : inst.delayEdges) {
                    int source = inst.inputPins[it.first.first];
                    int sink = inst.outputPins[it.first.second];
                    auto sourceNode = routetree.getTreeNodeByIdx(source);
                    auto sinkNode = routetree.getTreeNodeByIdx(sink);
                    if (sourceNode == nullptr || sinkNode == nullptr) continue;
                    if (sourceNode->net == nullptr || sinkNode->net == nullptr) continue;
                    inst.outputsAT[it.first.second] = std::max(inst.outputsAT[it.first.second], inst.inputsAT[it.first.first] + it.second);
                }

            for (int i = 0; i < inst.outputsAT.size(); i++) {
                auto node = routetree.getTreeNodeByIdx(inst.outputPins[i]);
                if (node == nullptr) continue;
                auto net = routetree.getTreeNodeByIdx(inst.outputPins[i])->net;
                if (net == nullptr) continue;
                for (auto netSink : net->getSinks()) {
                    auto& sinkInst = insts[routetree.routeGraph()->getVertexInst(netSink)];
                    for (int j = 0; j < sinkInst.inputPins.size(); j++) {
                        if (sinkInst.inputPins[j] == netSink) {
                            sinkInst.inputsAT[j] = std::max(sinkInst.inputsAT[j], inst.outputsAT[i] + routetree.getTreeNodeByIdx(netSink)->nodeDelay);
                            if (!sinkInst.terminal)
                                instDelay[sinkInst.instId] = std::max(instDelay[sinkInst.instId], sinkInst.inputsAT[j]);
                            break;
                        }
                    }
                    visited[sinkInst.instId]++;
                    if (visited[sinkInst.instId] == sinkInst.inputNetCnt && !sinkInst.terminal) {
                        q.push(sinkInst.instId);
                    }
                }
            }
        }


//Calculate RAT
        visited.assign(insts.size(), 0);
        for (auto& inst : insts) {
            if (inst.outputNetCnt == 0 || inst.terminal) {
                q.push(inst.instId);
            }
            instRAT[inst.instId] = period;
        }

        std::ofstream ofs("rat.debug");
        while (!q.empty()) {
            auto& inst = insts[q.front()];
            q.pop();

            // std::cout << inst.name << ' ' << instRAT[inst.instId] << std::endl;
            // getchar();

            // for (auto rat : inst.outputsRAT) instRAT[inst.instId] = std::min(instRAT[inst.instId], rat);


            for (int i = 0; i < inst.inputsRAT.size(); i++)
                inst.inputsRAT[i] = instRAT[inst.instId];
            
            if (!inst.terminal)
                for (auto& it : inst.delayEdges) {
                    int source = inst.inputPins[it.first.first];
                    int sink = inst.outputPins[it.first.second];
                    auto sourceNode = routetree.getTreeNodeByIdx(source);
                    auto sinkNode = routetree.getTreeNodeByIdx(sink);
                    if (sourceNode == nullptr || sinkNode == nullptr) continue;
                    if (sourceNode->net == nullptr || sinkNode->net == nullptr) continue;
                    inst.inputsRAT[it.first.first] = std::min(inst.inputsRAT[it.first.first], inst.outputsRAT[it.first.second] - it.second);
                }

            for (int i = 0; i < inst.inputsRAT.size(); i++) {
                auto node = routetree.getTreeNodeByIdx(inst.inputPins[i]);
                if (node == nullptr) continue;
                auto net = routetree.getTreeNodeByIdx(inst.inputPins[i])->net;
                if (net == nullptr) continue;
                int netSource = net->getSource();
                auto& sourceInst = insts[routetree.routeGraph()->getVertexInst(netSource)];
                for (int j = 0; j < sourceInst.outputPins.size(); j++) {
                    if (sourceInst.outputPins[j] == netSource) {
                        sourceInst.outputsRAT[j] = std::min(sourceInst.outputsRAT[j], inst.inputsRAT[i] - routetree.getTreeNodeByIdx(inst.inputPins[i])->nodeDelay);
                        if (!sourceInst.terminal)
                            instRAT[sourceInst.instId] = std::min(instRAT[sourceInst.instId], sourceInst.outputsRAT[j]);
                        break;
                    }
                }
                visited[sourceInst.instId]++;
                // std::cout << visited[sourceInst.instId] << ' ' << sourceInst.outputNetCnt << std::endl;
                // getchar();
                ofs << "from " << inst.name << " to " << sourceInst.name << "(" << sourceInst.terminal <<  "), visited: " << visited[sourceInst.instId] << ", output connect: " << sourceInst.outputNetCnt << std::endl;
                if (visited[sourceInst.instId] == sourceInst.outputNetCnt && !sourceInst.terminal) {
                    ofs << "adding " << sourceInst.name << std::endl;
                    q.push(sourceInst.instId);
                }
                // }
            }
        }
        ofs.close();

        for (auto& inst : insts) {
            inst.calcSlack();
            for (auto is : inst.inputsSlack) {
                if (is < 0) {
                    wns = std::min(wns, is);
                    tns += is;
                }
            }
            for (auto os : inst.outputsSlack) {
                if (os < 0) {
                    wns = std::min(wns, os);
                    tns += os;
                }
            }
        }

    }

    void InstList::printSTA() {
        std::ofstream ofs("sta.out");
        for (auto inst : insts) {
            ofs << inst.name << ' ' << inst.instId << ' ' << inst.inputNetCnt << ' ' << inst.outputNetCnt << std::endl;
            for (int i = 0; i < inst.inputPins.size(); i++) 
                ofs << "Input Pin " << inst.inputPins[i] << " AT: " << inst.inputsAT[i] << " RAT: " << inst.inputsRAT[i] << std::endl;
            for (int i = 0; i < inst.outputPins.size(); i++) 
                ofs << "Output Pin " << inst.outputPins[i] << " AT: " << inst.outputsAT[i] << " RAT: " << inst.outputsRAT[i] << std::endl;
            
        }
    }
    void Inst::calcSlack() {
        for (int i = 0; i < inputPins.size(); i++)
            inputsSlack[i] = inputsRAT[i] - inputsAT[i];
        for (int i = 0; i < outputPins.size(); i++)
            outputsSlack[i] = outputsRAT[i] - outputsAT[i];
    }

}