#include "predictmap.h"
#include "path.h"

#include <queue>
#include <set>
#include <chrono>
#include <future>

namespace router {
    void PredictMap::initPredictMap(std::shared_ptr<database::GridLayout> layout) {
        using namespace std::chrono;
        high_resolution_clock::time_point build_s, build_e;
        build_s = high_resolution_clock::now();
        std::cout << "Start Building Predict Map" << std::endl;

        auto& lib = layout->getModuleLibrary();
        int totalPins = 0;
        for (auto it : lib) {
            auto topModule = it.second;
            std::queue<std::shared_ptr<database::Module> > moduleQueue;
            moduleQueue.push(topModule);
            while (!moduleQueue.empty()) {
                auto now = moduleQueue.front();
                moduleQueue.pop();

                auto &ports = now->allPorts();
                for (auto port : ports) {
                    int width = port.second->getWidth();
                    for (int i = 0; i < width; i++) {
                        pinIdx[port.second->getPinByIdx(i)] = totalPins++;
                    }
                }

                auto &subModules = now->allSubmodules();
                for (auto subModule : subModules) {
                    moduleQueue.push(subModule.second);
                }
            }
            std::cout << topModule->getName() << ' ' << totalPins << std::endl;
        }

        sameGridDist.resize(totalPins);
        diffGridDist.resize(totalPins);
        for (int i = 0; i < totalPins; i++) {
            sameGridDist[i].resize(totalPins, std::numeric_limits<COST_T>::max());
            diffGridDist[i].resize(totalPins, std::numeric_limits<COST_T>::max());
        }

        std::cout << "total Pins: " << totalPins << std::endl;
        std::vector<COST_T> dist(graph->getVertexNum());
        int graphWidth = graph->getWidth();
        int graphHeight = graph->getHeight();

        std::set<std::shared_ptr<database::Module>> visitedModules;
        for (int i = 0; i < graphWidth; i++)
            for (int j = 0; j < graphHeight; j++) {
                auto gridModule = layout->getContent(i, j).gridModule;
                if (visitedModules.find(gridModule) != visitedModules.end()) continue;
                visitedModules.insert(gridModule);
                int pinCnt = graph->vertexId[i * graphHeight + j].size();
                for (int id = 0; id < pinCnt; id++) {
                    int source = graph->vertexId[i * graphHeight + j][id];
                    std::shared_ptr<database::Pin> pin = graph->getVertexByIdx(source);
                    // std::cout << gridModule->getName() << ' ' << pin->getName() << std::endl;
                    int pinId = pinIdx[pin];
                    dist.assign(graph->getVertexNum(), std::numeric_limits<COST_T>::max());
                    std::priority_queue<std::shared_ptr<PathNode>> q;
                    q.push(std::make_shared<PathNode>(source));
                    INDEX_T sourceX = graph->getPosHigh(source).X();
                    INDEX_T sourceY = graph->getPosHigh(source).Y();
                    dist[source] = 0;

                    std::vector<bool> visitedPin(totalPins, false);

                    while (!q.empty()) {
                        auto now = q.top();
                        q.pop();

                        int nowVertex = now->getHeadPin();
                        auto nowPin = graph->getVertexByIdx(now->getHeadPin());
                        int nowPinId = pinIdx[nowPin];
                        INDEX_T nowX = graph->getPosHigh(nowVertex).X();
                        INDEX_T nowY = graph->getPosHigh(nowVertex).Y();

                        if (nowX == sourceX && nowY == sourceY) {
                            sameGridDist[pinId][nowPinId] = now->getCost();
                        }
                        else {
                            if (visitedPin[nowPinId]) continue;
                            visitedPin[nowPinId] = true;
                            diffGridDist[pinId][nowPinId] = now->getCost() - abs(nowX - sourceX) - abs(nowY - sourceY);
                        }

                        int vertexDegree = graph->getVertexDegree(nowVertex);
                        for (int k = 0; k < vertexDegree; k++) {
                            int nexVertex = graph->getEdge(nowVertex, k);
                            COST_T edgeCost = graph->getEdgeCost(nowVertex, k);
                            if (dist[nexVertex] > dist[nowVertex] + edgeCost) {
                                dist[nexVertex] = dist[nowVertex] + edgeCost;
                                q.push(std::make_shared<PathNode>(nexVertex, nullptr, dist[nowVertex] + edgeCost));
                            }
                        }
                    }
                }
            }

        build_e = high_resolution_clock::now();
        duration<double, std::ratio<1, 1000>> duration_ms(build_e - build_s);
        std::cout << "Build Predict Map cost " << duration_ms.count() << "ms" << std::endl;
    }

    COST_T PredictMap::predictDist(int source, int sink) {
        INDEX_T sourceX = graph->getPosHigh(source).X();
        INDEX_T sourceY = graph->getPosHigh(source).Y();
        int sorucePinId = pinIdx[graph->getVertexByIdx(source)];

        INDEX_T sinkX = graph->getPosHigh(sink).X();
        INDEX_T sinkY = graph->getPosHigh(sink).Y();
        int sinkPinId = pinIdx[graph->getVertexByIdx(sink)];

        if (sourceX == sinkX && sourceY == sinkY) {
            return sameGridDist[sorucePinId][sinkPinId];
        }
        else {
            return abs(sourceX - sinkX) + abs(sourceY -sinkY) + diffGridDist[sorucePinId][sinkPinId];
        }
    }
}