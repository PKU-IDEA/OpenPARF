#include "router.h"
#include "utils/printer.h"
#include "pathfinderastar.h"
#include "utils/printer.h"
#include <iostream>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <queue>
#include <future>
#include <algorithm>
// #include "tilerouter.h"
// #include "localrouter.h"
using namespace std::chrono;
namespace router {


void Router::routeSingleNet(std::shared_ptr<Net> net) {
    auto& routetree = Pathfinder::routetree;
    net->inQueueCnt = net->outQueueCnt = 0;
    if (net->getRouteStatus() == UNROUTED || routetree.ripupDfsSearch(routetree.getNetRoot(net), 0, 0, net)) {
        net->setRouteStatus(UNROUTED);
        AStarPathfinder finder(net, graph);
        RouteStatus status = finder.run();
        net->setRouteStatus(status);
    }
}

void Router::runTaskflow() {
    high_resolution_clock::time_point dr_s, dr_e;
    dr_s = high_resolution_clock::now();
    int maxRipupIter = 1000;
    double totalDRTime = 0;
    for (int i = 0; i < netlist.size(); i++)
        netlist[i]->netId = i;
    Pathfinder::routetree.init(graph, netlist);
    std::cout << "Mem Peak: " << get_memory_peak() << "M\n";

    RouteGraph::presFac = RouteGraph::initialPresentCongestionFactor;

    auto cmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> &b) {
        auto guide_a = a->getGuide();
        auto guide_b = b->getGuide();
        return (guide_a.end_x - guide_a.start_x + 1) * (guide_a.end_y - guide_a.start_y + 1)
            <  (guide_b.end_x - guide_b.start_x + 1) * (guide_b.end_y - guide_b.start_y + 1);
    };
    sort(netlist.begin(), netlist.end(), cmp);
    
    tf::Taskflow taskflow;
    std::vector<tf::Task> tasks(netlist.size());
    int totalNets = netlist.size();
    high_resolution_clock::time_point sch_s, sch_e;
    sch_s = high_resolution_clock::now();
    std::vector<std::vector<int>> nets_id(graph->getWidth());
    for (int i = 0; i < graph->getWidth(); i++) nets_id[i].assign(graph->getHeight(), -1);
    for (int i = 0; i < totalNets; i++) {
        auto net = netlist[i];
        tasks[i] = taskflow.emplace([net, this](){ routeSingleNet(net); });
        auto guide = netlist[i]->getGuide();
        for (int ii = std::max(0, guide.start_x); ii <= guide.end_x && ii < graph->getWidth(); ii++)
            for (int jj = std::max(0, guide.start_y); jj <= guide.end_y && jj < graph->getHeight(); jj++) {
                if (nets_id[ii][jj] != -1) tasks[i].succeed(tasks[nets_id[ii][jj]]);
                nets_id[ii][jj] = i;
            }
    }
    sch_e = high_resolution_clock::now();
    duration<double, std::ratio<1, 1> > duration_sch(sch_e - sch_s);
    std::cout << "Scheduling Runtime: " << duration_sch.count() << "s" << std::endl; 

    for (int iter = 0; iter < maxRipupIter; iter++) {
        std::cout << "----------- Iter #" << iter << " -----------" << std::endl;
        Pathfinder::iter = iter;
        high_resolution_clock::time_point route_s, route_e;
        route_s = high_resolution_clock::now();

        int vertexNum = graph->getVertexNum();

        Pathfinder::visited.assign(vertexNum, -1);
        Pathfinder::treeNodes.assign(vertexNum, nullptr);
        Pathfinder::cost.assign(vertexNum, std::numeric_limits<COST_T>::max());
        Pathfinder::prev.assign(vertexNum, -1);
        graph->updateVertexCost();
        
        tf::Executor executor(32);
        executor.run(taskflow).wait();

        route_e = high_resolution_clock::now();
        duration<double, std::ratio<1, 1>> duration_s(route_e - route_s);
        totalDRTime += duration_s.count();
        int64_t totalInQueueCnt = 0, totalOutQueueCnt = 0;
        for (auto net : netlist) {
            totalInQueueCnt += net->inQueueCnt;
            totalOutQueueCnt += net->outQueueCnt;
        }


        std::cout << "InQueueCnt: " << totalInQueueCnt << " OutQueueCnt: " << totalOutQueueCnt << std::endl;
        printf("Iter #%d, Runtime: %lfs\n", iter, duration_s.count());
        COST_T totalWL = Pathfinder::routetree.getTotalWL();
        std::cout << "WL: " << totalWL << std::endl;
        std::cout << "presFac: " << RouteGraph::presFac << std::endl;
        RouteGraph::presFac = std::min(RouteGraph::presentCongestionMultiplier * RouteGraph::presFac, RouteGraph::maxPresentCongestionFactor);
        if (totalOutQueueCnt == 0) break;
    } 
    dr_e = high_resolution_clock::now();
    duration<double, std::ratio<1, 1> > duration_dr(dr_e - dr_s);
    std::cout << "DR Runtime: " << duration_dr.count() << "s" << std::endl; 

}

}
