#ifndef TIMER_H_
#define TIMER_H_

#include "timinggraph.h"
#include "routegraph.h"
#include "net.h"
namespace router
{

class Timer {
public:
    Timer(){}
    Timer(std::shared_ptr<RouteGraph> graph) : routegraph(graph) {}

    void buildTimingGraph(std::vector<std::shared_ptr<Net>>& netlist);
    void STA();
    void STAAndReportCriticalPath();
    void estimateSTA();
    void updatePinCritical(std::vector<std::shared_ptr<Net>>& netlist);
    
    COST_T estimateEdgeDelay(TimingEdge& edge) {
        if (edge.edgeType == INSTEDGE) return edge.edgeDelay;
        int source = timinggraph.getVertex(edge.source).vertexIdx;
        int sink = timinggraph.getVertex(edge.sink).vertexIdx;
        int sourceX = routegraph->getPos(source).X(), sourceY = routegraph->getPos(source).Y();
        int sinkX = routegraph->getPos(sink).X(), sinkY = routegraph->getPos(sink).Y();
        int dx = abs(sourceX - sinkX), dy = abs(sourceY - sinkY);
        COST_T costX = (dx / 6) * 120 + ((dx % 6) / 2) * 60 + (dx % 2) * 50;
        COST_T costY = (dy / 6) * 120 + ((dy % 6) / 2) * 60 + (dy % 2) * 50;
        if (dx == 0 && dy == 0) return 0;
        return costX + costY + 100;
    }

    void printSTA();
    void printEdgeDelay();

private:
    TimingGraph timinggraph;
    std::shared_ptr<RouteGraph> routegraph;

    std::vector<COST_T> AT;
    std::vector<COST_T> RAT;
    std::vector<COST_T> slack;

    std::unordered_map<int, int> timinggraphId;
};

} // namespace router


#endif //TIMER_H_