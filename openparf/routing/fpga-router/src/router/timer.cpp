#include "timer.h"
#include <queue>
#include <iostream>
#include <fstream>
namespace router {
    void Timer::buildTimingGraph(std::vector<std::shared_ptr<Net>>& netlist) {
        for (auto net : netlist) {
            int source = net->getSource();
            timinggraphId[source] = timinggraph.addVertex(source);
            for (auto sink : net->getSinks()) {
                timinggraphId[sink] = timinggraph.addVertex(sink);
                timinggraph.addEdge(timinggraphId[source], timinggraphId[sink], NETEDGE, -1);
            }
        }
        auto instlist = routegraph->getInstList().getInsts();
        for (auto& inst : instlist) {
            for (auto& edge : inst.getDelayEdges()) {
                int edgeSource = inst.getInputPins()[edge.first.first];
                int edgeSink = inst.getOutputPins()[edge.first.second];
                if (timinggraphId.find(edgeSource) == timinggraphId.end() || timinggraphId.find(edgeSink) == timinggraphId.end()) continue;
                timinggraph.addEdge(timinggraphId[edgeSource], timinggraphId[edgeSink], INSTEDGE, edge.second);
            }
        }
    }

    void Timer::STA() {
        int vertexNum = timinggraph.getVertexNum();
        std::vector<int> visited(vertexNum, 0);
        std::queue<int> q;

        AT.assign(vertexNum, 0);
        RAT.assign(vertexNum, std::numeric_limits<COST_T>::max());
        slack.assign(vertexNum, std::numeric_limits<COST_T>::max());

        for (int i = 0; i < vertexNum; i++) {
            if (timinggraph.getVertex(i).inputDegree == 0)
                q.push(i);
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSource; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sourcePrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int to = edge.sink;
                visited[to]++;
                AT[to] = std::max(AT[to], AT[now] + timinggraph.getEdgeDelay(edge));
                if (visited[to] == timinggraph.getVertex(to).inputDegree)
                    q.push(to);
            }
        }
        
        visited.assign(vertexNum, 0);
        for (int i = 0; i < vertexNum; i++) {
            auto pin = routegraph->getVertexByIdx(timinggraph.getVertex(i).vertexIdx);
            if (timinggraph.getVertex(i).outputDegree == 0 && pin->getName().find("SR") == std::string::npos) {
                q.push(i);
                RAT[i] = InstList::period;
            }
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSink; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sinkPrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int from = edge.source;
                visited[from]++;
                RAT[from] = std::min(RAT[from], RAT[now] - timinggraph.getEdgeDelay(edge));
                if (visited[from] == timinggraph.getVertex(from).outputDegree)
                    q.push(from);
            }
        }

        COST_T wns = 0, tns = 0;
        for (int i = 0; i < vertexNum; i++) {
            slack[i] = RAT[i] - AT[i];
            routegraph->setVertexSlack(timinggraph.getVertex(i).vertexIdx, slack[i]);
            if (slack[i] < 0) {
                wns = std::min(wns, slack[i]);
                if (timinggraph.getVertex(i).outputDegree == 0)
                    tns += slack[i];
            }
        }
        std::cout << "Timer result: WNS: " << wns << " TNS: " << tns << std::endl;
    }
    

    void Timer::STAAndReportCriticalPath() {
        int vertexNum = timinggraph.getVertexNum();
        std::vector<int> visited(vertexNum, 0);
        std::vector<int> prev(vertexNum, -1);
        std::queue<int> q;

        AT.assign(vertexNum, 0);
        RAT.assign(vertexNum, std::numeric_limits<COST_T>::max());
        slack.assign(vertexNum, std::numeric_limits<COST_T>::max());

        for (int i = 0; i < vertexNum; i++) {
            if (timinggraph.getVertex(i).inputDegree == 0)
                q.push(i);
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSource; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sourcePrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int to = edge.sink;
                visited[to]++;
                if (AT[now] + timinggraph.getEdgeDelay(edge) > AT[to]) {
                    AT[to] = AT[now] + timinggraph.getEdgeDelay(edge);
                    prev[to] = now;
                }
                if (visited[to] == timinggraph.getVertex(to).inputDegree)
                    q.push(to);
            }
        }
        
        visited.assign(vertexNum, 0);
        int maxId; COST_T maxDelay = 0;
        for (int i = 0; i < vertexNum; i++) {
            auto pin = routegraph->getVertexByIdx(timinggraph.getVertex(i).vertexIdx);
            if (timinggraph.getVertex(i).outputDegree == 0 && pin->getName().find("SR") == std::string::npos) {
                q.push(i);
                RAT[i] = InstList::period;
                if (AT[i] > maxDelay) {
                    maxDelay = AT[i];
                    maxId = i;
                }
            }
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSink; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sinkPrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int from = edge.source;
                visited[from]++;
                RAT[from] = std::min(RAT[from], RAT[now] - timinggraph.getEdgeDelay(edge));
                if (visited[from] == timinggraph.getVertex(from).outputDegree)
                    q.push(from);
            }
        }

        COST_T wns = 0, tns = 0;
        for (int i = 0; i < vertexNum; i++) {
            slack[i] = RAT[i] - AT[i];
            routegraph->setVertexSlack(timinggraph.getVertex(i).vertexIdx, slack[i]);
            if (slack[i] < 0) {
                wns = std::min(wns, slack[i]);
                if (timinggraph.getVertex(i).outputDegree == 0)
                    tns += slack[i];
            }
        }
        std::cout << "Timer result: WNS: " << wns << " TNS: " << tns << std::endl;
        std::cout << "reporting critcal path:" << std::endl;
        auto& instlist = routegraph->getInstList();
        int now = maxId;
        while (now != -1) {
            int vertex = timinggraph.getVertex(now).vertexIdx;
            auto &inst = instlist.getInsts()[routegraph->getVertexInst(vertex)];
            std::cout << inst.getName() << ' ' << routegraph->getVertexByIdx(vertex)->getName() << ' ' << routegraph->getPos(vertex).X() << ' ' << routegraph->getPos(vertex).Y() << " " << "AT: " << AT[now] << " RAT: " << RAT[now] << " slack: " << slack[now] << std::endl;
            now = prev[now];
        }
    }

    void Timer::estimateSTA() {
        int vertexNum = timinggraph.getVertexNum();
        std::vector<int> visited(vertexNum, 0);
        std::queue<int> q;
        std::vector<int> prev(vertexNum, -1);

        AT.assign(vertexNum, 0);
        RAT.assign(vertexNum, std::numeric_limits<COST_T>::max());
        slack.assign(vertexNum, 0);

        for (int i = 0; i < vertexNum; i++) {
            if (timinggraph.getVertex(i).inputDegree == 0)
                q.push(i);
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSource; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sourcePrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int to = edge.sink;
                visited[to]++;
                if (AT[now] + estimateEdgeDelay(edge) > AT[to]) {
                    AT[to] = AT[now] + estimateEdgeDelay(edge);
                    prev[to] = now;
                }
                if (visited[to] == timinggraph.getVertex(to).inputDegree)
                    q.push(to);
            }
        }
        
        visited.assign(vertexNum, 0);
        int maxId; COST_T maxDelay = 0;
        for (int i = 0; i < vertexNum; i++) {
            auto pin = routegraph->getVertexByIdx(timinggraph.getVertex(i).vertexIdx);
            if (timinggraph.getVertex(i).outputDegree == 0 && pin->getName().find("SR") == std::string::npos) {
                q.push(i);
                RAT[i] = InstList::period;
                if (AT[i] > maxDelay) {
                    maxDelay = AT[i];
                    maxId = i;
                }
            }
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();

            for (int edgeId = timinggraph.getVertex(now).headSink; edgeId != -1; edgeId = timinggraph.getEdge(edgeId).sinkPrev) {
                auto& edge = timinggraph.getEdge(edgeId);
                int from = edge.source;
                visited[from]++;
                RAT[from] = std::min(RAT[from], RAT[now] - estimateEdgeDelay(edge));
                if (visited[from] == timinggraph.getVertex(from).outputDegree)
                    q.push(from);
            }
        }

        COST_T wns = 0, tns = 0;
        for (int i = 0; i < vertexNum; i++) {
            slack[i] = RAT[i] - AT[i];
            routegraph->setVertexSlack(timinggraph.getVertex(i).vertexIdx, slack[i]);
            if (slack[i] < 0) {
                wns = std::min(wns, slack[i]);
                if (timinggraph.getVertex(i).outputDegree == 0)
                    tns += slack[i];
            }
        }
        std::cout << "Estimate result: WNS: " << wns << " TNS: " << tns << std::endl;
        auto& instlist = routegraph->getInstList();
        int now = maxId;
        while (now != -1) {
            int vertex = timinggraph.getVertex(now).vertexIdx;
            auto &inst = instlist.getInsts()[routegraph->getVertexInst(vertex)];
            std::cout << inst.getName() << ' ' << routegraph->getVertexByIdx(vertex)->getName() << ' ' << routegraph->getPos(vertex).X() << ' ' << routegraph->getPos(vertex).Y() << " " << "AT: " << AT[now] << " RAT: " << RAT[now] << " slack: " << slack[now] << std::endl;
            now = prev[now];
        }
    }

    void Timer::updatePinCritical(std::vector<std::shared_ptr<Net>>& netlist) {
        for (auto net : netlist) {
            int sinkSize = net->getSinkSize();
            for (int i = 0; i < sinkSize; i++) {
                int sink = net->getSinkByIdx(i);
                int sinkTimingIdx = timinggraphId[sink];
                COST_T critical = 1.0 - slack[sinkTimingIdx] / RAT[sinkTimingIdx];
                critical = std::min(0.999f, std::max(0.0f, critical));
                net->setSinkCritical(i, critical);
            }
        }
    }

    void Timer::printSTA() {
        std::ofstream ofs("sta.out");

        int vertexNum = timinggraph.getVertexNum();
        for (int i = 0; i < vertexNum; i++) {
            auto& vertex = timinggraph.getVertex(i);
            int originIdx = vertex.vertexIdx;
            ofs << routegraph->getVertexByIdx(originIdx)->getName() << " AT: " << AT[i] << " RAT: " << RAT[i] << " slack: " << slack[i] << std::endl;
        } 

        ofs.close();
    }

    void Timer::printEdgeDelay() {
        std::ofstream ofs("edgedelay.out");
        auto& edges = timinggraph.getEdges();
        for (auto &edge : edges) {
            auto &source = timinggraph.getVertex(edge.source);
            auto &sink = timinggraph.getVertex(edge.sink);
            auto &instlist = routegraph->getInstList().getInsts();
            auto sourceInst = instlist[routegraph->getVertexInst(source.vertexIdx)].getName();
            auto sinkInst = instlist[routegraph->getVertexInst(sink.vertexIdx)].getName();
            auto sourcePin = routegraph->getVertexByIdx(source.vertexIdx)->getName();
            auto sinkPin = routegraph->getVertexByIdx(sink.vertexIdx)->getName();
            auto estimatedDelay = estimateEdgeDelay(edge);
            auto routedDelay = timinggraph.getEdgeDelay(edge);
            ofs << sourceInst << " " << sourcePin << " " << sinkInst << " " << sinkPin << " " << estimatedDelay << " " << routedDelay << std::endl;  
        }
        ofs.close();
    }
}