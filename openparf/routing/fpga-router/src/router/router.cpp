#include "router.h"
#include "test/netgenerator.h"
#include "utils/printer.h"
#include "pathfinderastar.h"
#include "scheduler.h"
#include "globalrouter.h"
#include "utils/vpr/vprparser.h"
#include "utils/printer.h"
#include <iostream>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <queue>
#include <future>
#include <algorithm>
#include "tilerouter.h"
#include "localrouter.h"
#include "timer.h"
// #include "netconnect.h"
using namespace std::chrono;
namespace router {

int Router::maxRipupIter;
int Router::printCongestMapIter;

Router::Router(std::shared_ptr<RouteGraph> _graph, std::shared_ptr<database::GridLayout> layout, std::string inNetFile, int mttype) {
    MTType = mttype;
    graph = _graph;
    std::unordered_map<std::string, std::shared_ptr<database::Pin>> pinMap;
    auto lib = layout->getModuleLibrary();
    for (auto libContent : lib) {
        buildPinMap(pinMap, libContent.second);
    }
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(inNetFile.c_str());
    pugi::xml_node netlistInfo = doc.child("netlist");
    for (auto netInfo : netlistInfo.children("net")) {
        std::string rs = netInfo.attribute("result").value();
        if (rs != "Unrouted") continue;
        std::shared_ptr<Net> net(new Net(netInfo.attribute("name").value()));
        auto sourceInfo = netInfo.child("source");
        std::string sourceName = sourceInfo.attribute("pin").value();
        int sourceX = sourceInfo.attribute("x").as_int();
        int sourceY = sourceInfo.attribute("y").as_int();
        net->setSource(graph->getVertexId(sourceX, sourceY, pinMap[sourceName]));
        net->addGuideNode(sourceX, sourceY);
        for (auto sinkInfo : netInfo.children("sink")) {
            std::string sinkName = sinkInfo.attribute("pin").value();
            int sinkX = sinkInfo.attribute("x").as_int();
            int sinkY = sinkInfo.attribute("y").as_int();
            net->addSink(graph->getVertexId(sinkX, sinkY, pinMap[sinkName]));
            net->addGuideNode(sinkX, sinkY);
        }
        netlist.push_back(net);
    }
}

void Router::buildPinMap(std::unordered_map<std::string, std::shared_ptr<database::Pin>>& pinMap, std::shared_ptr<database::Module> currentModule) {
    auto allPorts = currentModule->allPorts();
    for (auto it : allPorts) {
        int width = it.second->getWidth();
        for (int i = 0; i < width; i++) {
            auto pin = it.second->getPinByIdx(i);
            pinMap[pin->getName()] = pin;
        }
    }
    for (auto subModule : currentModule->allSubmodules()) {
        buildPinMap(pinMap, subModule.second);
    }
}

void Router::run() {
    high_resolution_clock::time_point gr_s, gr_e;
    gr_s = high_resolution_clock::now();
    GlobalRouter gRouter(graph);
    gRouter.run(netlist);
    gr_e = high_resolution_clock::now();
    duration<double, std::ratio<1, 1> > duration_gr(gr_e - gr_s);
    std::cout << "GR Runtime: " << duration_gr.count() << "s" << std::endl; 

    Timer timer(graph);
    if (InstList::period) {
        timer.buildTimingGraph(netlist);
    }
    // exit(-1);
    high_resolution_clock::time_point dr_s, dr_e;
    dr_s = high_resolution_clock::now();
    // int maxRipupIter = 311;
    // std::shared_ptr<RouteGraph> graph = builder.run();
    double totalDRTime = 0;
    for (int i = 0; i < netlist.size(); i++)
        netlist[i]->netId = i;
    Scheduler scheduler(graph);
    Pathfinder::routetree.init(graph, netlist);
    // routeTileNets();
    // exit(-1);
    std::cout << "total Pin Num: " << graph->getVertexNum() << std::endl;
    std::cout << "Mem Peak: " << get_memory_peak() << "M\n";

    RouteGraph::presFac = RouteGraph::presFacFirstIter;

    auto cmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> &b) {
                if ( a->getRerouteTime() == b->getRerouteTime()) {
                auto guide_a = a->getGuide();
                auto guide_b = b->getGuide();
                return (guide_a.end_x - guide_a.start_x + 1) * (guide_a.end_y - guide_a.start_y + 1)
                    >  (guide_b.end_x - guide_b.start_x + 1) * (guide_b.end_y - guide_b.start_y + 1);
                }
                return a->getRerouteTime() > b->getRerouteTime();
    };

    auto delayCmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> &b) {
            if (Pathfinder::isTimingDriven) {
                COST_T mca = 0, mcb = 0;
                for (auto sink : a->getSinks()) {
                    mca = std::max(mca, a->getSinkCritical(sink));
                }
                for (auto sink : b->getSinks()) {
                    mcb = std::max(mcb, b->getSinkCritical(sink));
                }
                if (mca == mcb) {
                    if ( a->getRerouteTime() == b->getRerouteTime()) {
                        auto guide_a = a->getGuide();
                        auto guide_b = b->getGuide();
                        return (guide_a.end_x - guide_a.start_x + 1) * (guide_a.end_y - guide_a.start_y + 1)
                            >  (guide_b.end_x - guide_b.start_x + 1) * (guide_b.end_y - guide_b.start_y + 1);
                    }
                    return a->getRerouteTime() > b->getRerouteTime();
                }
                return mca > mcb;
            }
    };

    Pathfinder::delay.assign(graph->getVertexNum(), 0);

    if (Pathfinder::isTimingDriven) {
        timer.estimateSTA();
        timer.updatePinCritical(netlist);
    } else if (InstList::period) {
        timer.estimateSTA();
    }

    for (int iter = 0; iter < maxRipupIter; iter++) {
        // if (iter >= 290)
        // RouteGraph::debugging = true;
        if (iter == printCongestMapIter)
            RouteGraph::dumpingCongestMap = true;
        else 
            RouteGraph::dumpingCongestMap = false;
        std::cout << "----------- Iter #" << iter << " -----------" << std::endl;
        Pathfinder::iter = iter;
        high_resolution_clock::time_point route_s, route_e;
        route_s = high_resolution_clock::now();

        Pathfinder::visited.assign(graph->getVertexNum(), -1);
        Pathfinder::treeNodes.assign(graph->getVertexNum(), nullptr);
        Pathfinder::cost.assign(graph->getVertexNum(), std::numeric_limits<COST_T>::max());
        Pathfinder::prev.assign(graph->getVertexNum(), -1);

        if(iter < 10) 
            sort(netlist.begin(), netlist.end(), cmp);
        else
            std::reverse(netlist.begin(), netlist.end()); 
        int vertexNum = graph->getVertexNum();
        Pathfinder::routetree.ripup(netlist, (iter > 0 && iter % 2 == 0));
        std::vector<std::shared_ptr<Net>> unroutedNets;
        int successCnt = 0, failedCnt = 0, unroutedCnt = 0;
        
        int bb_threshold[10] = {1, 5, 10, 50, 100, 200, 500, 1000, 2000, 1000000};
        int sinks_threshold[10] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 1000000};
        int bb_netcnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int sinks_netcnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double bb_rtcnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double sinks_rtcnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        // if (iter >= 30) std::reverse(netlist.begin(), netlist.end());

        // if (iter == 20) Pathfinder::routetree.congestAddCost = 0;
        for (auto net : netlist) {
            // if (net->getName() != "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs2_ff[32]") continue;
            // if (net->getName() != "sparc_exu:exu|bw_r_irf:irf|bw_r_irf_core:bw_r_irf_core|bw_r_irf_register:register16|window_rtl_0_bypass[1]")
            //     continue;
            // std::cout << net->guide.start_x << ' ' << net->guide.end_x << ' ' << net->guide.start_y << ' ' << net->guide.end_y << std::endl;
            // if (iter == 30) net->useGlobalResult(false);
            // std::cout << net->getName() << ' ' << (net->getRouteStatus() == SUCCESS ? "SUCCESS" : "CONGEST") << std::endl;
            if (iter == 300) net->useGlobalResult(false);
            if (net->getRouteStatus() == SUCCESS)
                successCnt++;
            if (net->getRouteStatus() == FAILED) {
                // std::cout << net->getName() << std::endl;
                // if (iter == 1) {
                //     net->useGlobalResult(false);
                //     net->setRouteStatus(UNROUTED);
                // }
                // else 
                failedCnt++;
            }
            if (net->getRouteStatus() == UNROUTED || net->getRouteStatus() == CONGESTED) {
                net->setRouteStatus(UNROUTED);
                if (iter % 3 == 0 && iter > 0 && iter < 30 && net->useGlobalResult() == false)
                    net->expandGuide();
                if (MTType == SingleThread) {
                    auto rs = high_resolution_clock::now();
                    AStarPathfinder finder(net, graph);
                    RouteStatus status = finder.run();
                    // getchar();
                    net->setRouteStatus(status);
                    auto rt = high_resolution_clock::now();
                    duration<double, std::ratio<1, 1000>> duration_ms(rt - rs);
                    int num_sinks = net->getSinkSize();
                    int id = 0;
                    for (int i = 0; i < 10; i++) {
                        if (num_sinks <= sinks_threshold[i]) {
                            id = i;
                            break;
                        }
                    }
                    sinks_netcnt[id]++;
                    sinks_rtcnt[id] += duration_ms.count();
                    if (duration_ms.count() > 1000) {
                        std::cout << "NET " << net->getName() << " RT: " << duration_ms.count() << "ms\n";
                    }
                    // printf("Runtime: %lfms\n", duration_ms.count());
                }
                unroutedNets.push_back(net);
                unroutedCnt++;
            }
/* debugging */
// if (net->getName() == "sparc_mul_top:mul|sparc_mul_dp:dpath|mul64:mulcore|rs1_ff[0]") {
//     auto head = Pathfinder::routetree.getNetRoot(net);
//     std::queue<std::shared_ptr<TreeNode>> q;
//     q.push(head);
//     while (!q.empty()) {
//         auto now = q.front();
//         q.pop();
//         for (auto child = now->firstChild; child != nullptr; child = child->right) {
//             std::cout << now->nodeId << "->" << child->nodeId << std::endl;
//             q.push(child);
//         }
//     }
// }
/* end */
        }
        for (int i = 0; i < 10; i++) {
            printf("Sinks num from %d to %d, net count: %d, total RT: %lf\n", i ? sinks_threshold[i - 1] : 0, sinks_threshold[i], sinks_netcnt[i], sinks_rtcnt[i]);
        }
        std::cout << "Success : Failed : Unrouted " << successCnt << ':' << failedCnt << ':' << unroutedCnt << std::endl;
        if (totalDRTime >= 86400) break;
        // if (iter == 15) exit(-1);
        // getchar();
        // if (finished) break;
        if (unroutedNets.size() == 0) break;
        if (unroutedCnt <= 100 && MTType) {
            for (auto net : unroutedNets)
                std::cout << net->getName() << ' ';
            std::cout << std::endl;
            if (iter & 1)
            for (int i = unroutedCnt - 1; i >= 0; i--) {
                AStarPathfinder finder(unroutedNets[i], graph);
                RouteStatus status = finder.run();
                unroutedNets[i]->setRouteStatus(status);
            }
            else 
            for (int i = 0; i < unroutedCnt; i++) {
                AStarPathfinder finder(unroutedNets[i], graph);
                RouteStatus status = finder.run();
                unroutedNets[i]->setRouteStatus(status);
            }
        }
        else if (MTType == StaticSchedule) { 
            auto batches = scheduler.schedule(unroutedNets);
            std::cout << "total Batches: " << batches.size() << std::endl;
            omp_set_num_threads(40);
            int batchId = 0;
            for (auto batch : batches) {
                high_resolution_clock::time_point route_s, route_e;
                route_s = high_resolution_clock::now();
                int batchSize = batch.size();
#pragma omp parallel for
                for (int i = 0; i < batchSize; i++) {
                    AStarPathfinder finder(unroutedNets[batch[i]], graph);
                    RouteStatus status = finder.run();
                    unroutedNets[batch[i]]->setRouteStatus(status);
                }
                route_e = high_resolution_clock::now();
                duration<double, std::ratio<1, 1000>> duration_ms(route_e - route_s);
                int maxBBox = 0, maxDegree = 0;
                for (auto net : batch) {
                    auto guide = unroutedNets[net]->getGuide();
                    maxBBox = std::max(maxBBox, (guide.end_x - guide.start_x + 1) * (guide.end_y - guide.start_y + 1));
                    
                    maxDegree = std::max(maxDegree, unroutedNets[net]->getSinkSize());
                }
                if (get_memory_peak() >= 65536 * 4) {
                    printf("[Error] memory to LARGE!\n");
                    exit(-1);
                }
                // printf("Batch #%d, Runtime: %lfms\n, Batch Size: %d, Max BoundingBox Area:%d\n, maxDegree: %d\n",
                //         batchId++, duration_ms.count(), batchSize, maxBBox, maxDegree);
            }
        }
        else if (MTType == DynamicSchedule) {
            using namespace std::chrono;
            high_resolution_clock::time_point route_s, route_e;
            route_s = high_resolution_clock::now();
            scheduler.taskflowSchedule(unroutedNets);
            route_e = high_resolution_clock::now();
            duration<double, std::ratio<1, 1000>> duration_ms(route_e - route_s);
            printf("Runtime: %lfms\n", duration_ms.count());
        }
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
        // printVPRWirelength(netlist, graph);
        std::cout << "WL: " << Pathfinder::routetree.getTotalWL() << std::endl;

        std::cout << "presFac: " << RouteGraph::presFac << std::endl;
        std::cout << "Routed Sinks: " << Pathfinder::routedSinks << std::endl;
        if (iter == 0) RouteGraph::presFac = RouteGraph::presFacInit;
        else RouteGraph::presFac = std::min(RouteGraph::presFacMult * RouteGraph::presFac, (COST_T)1e25);
        // PathNode::astarFac = std::min(RouteGraph::presFacMult * PathNode::astarFac, (COST_T)1e20);
    
        // if (iter >= 290) {
        //     std::string outputDoc = "debug_result_" + std::to_string(iter);
        //     printRouteResult(netlist, outputDoc, graph);
        // }
        if (Pathfinder::isTimingDriven) {
            timer.STA();
            timer.updatePinCritical(netlist);
        } 
        else if (InstList::period) {
            timer.STA();
        }

        // std::cout << Pathfinder::routetree.getTreeNodeByIdx
         
    }
    dr_e = high_resolution_clock::now();
    duration<double, std::ratio<1, 1> > duration_dr(dr_e - dr_s);
    std::cout << "DR Runtime: " << duration_dr.count() << "s" << std::endl; 

    // graph->getInstList().calcDelayAndSlack();
    // graph->getInstList().printSTA();
    if (InstList::period) {
        timer.STAAndReportCriticalPath();
        timer.printSTA();
        timer.printEdgeDelay();
    }

    // double totalDumpGraphTime = 0, totalPathfindTime = 0;
    // for (auto net : netlist) {
    //     totalDumpGraphTime += net->dumpGraphTime;
    //     totalPathfindTime += net->pathfindTime;
    // }
    // std::cout << "Dump Graph Time : " << totalDumpGraphTime << " Pathfinding Time: " << totalPathfindTime << std::endl;
}

void Router::routeTileNets() {
    TileRouter tileRouter(graph, layout_);
    tileRouter.setRouter(this);
    tileRouter.tileRoute();
    // for (auto net : netlist) {
    //     int source = net->getSource();
    //     int sourceX = graph->getPos(source).X(), sourceY = graph->getPos(source).Y();
    //     int sinkSize = net->getSinkSize();
    //     for ()
    // }


}


}
