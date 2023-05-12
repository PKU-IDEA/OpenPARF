#include "scheduler.h"
#include "taskflow/taskflow.hpp"

#include <algorithm>
#include <array>
namespace router
{
    std::vector<std::vector<int> >& Scheduler::schedule(std::vector<std::shared_ptr<Net>>& netlist) {

        auto cmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> & b) {
            // if (a->getSinkSize() != b->getSinkSize()) return a->getSinkSize() > b->getSinkSize();
            auto guide_a = a->getGuide();
            auto guide_b = b->getGuide();
            return (guide_a.end_x - guide_a.start_x + 1) * (guide_a.end_y - guide_a.start_y + 1)
                >  (guide_b.end_x - guide_b.start_x + 1) * (guide_b.end_y - guide_b.start_y + 1) ;
        };

        std::sort(netlist.begin(), netlist.end(), cmp);
        int netlistSize = netlist.size();
        std::vector<bool> scheduled(netlistSize, false);
        batches.clear();

        // int height = graph->getHeight();

        // for (int batchId = 0; ; batchId++) {
        //     bool finished = true;
        //     std::vector<bool> vis(graph->getWidth(), graph->getHeight());
        //     batches.emplace_back();
        //     for (int i = 0; i < netlistSize; i++) {
        //         if (!scheduled[i] && netlist[i]->useGlobalResult()) {
        //             finished = false;
        //             bool flag = true;
        //             auto& gr = netlist[i]->getGlobalRouteResult();
        //             for (auto it : gr) {
        //                 if (vis[it.X() * height + it.Y()]) {
        //                     flag = false;
        //                     break;
        //                 }
        //             }
        //             if (flag) {
        //                 for (auto it : gr) vis[it.X() * height + it.Y()] = true;
        //                 batches[batchId].push_back(i);
        //                 scheduled[i] = true;
        //             }
        //         }
        //     }
        //     if (finished) break;
        // }
        // batches.pop_back();

        for (int batchId = 0;;batchId++) {
            // std::cout << "Scheduling Bat/ch# " << batchId << std::endl;
            RTree rtree;
            bool finished = true;
            batches.emplace_back();
            for (int i = 0; i < netlistSize; i++) {
                if (!scheduled[i]) {
                    finished = false;
                    auto guide = netlist[i]->getGuide();
                    boostBox box(boostPoint(guide.start_x, guide.start_y),
                                 boostPoint(guide.end_x, guide.end_y));
                    std::vector<std::pair<boostBox, int>> results;
                    rtree.query(bgi::intersects(box), std::back_inserter(results));
                    if (results.size() == 0) {
                        rtree.insert({box, i});
                        batches[batchId].push_back(i);
                        scheduled[i] = true;
                    }
                }
            }
            if (finished) break;
        }
        batches.pop_back();
        std::reverse(batches.begin(), batches.end());
        return batches;
    }

    void Scheduler::taskflowSchedule(std::vector<std::shared_ptr<Net>>& netlist) {
        tf::Taskflow taskflow;
        auto cmp = [](const std::shared_ptr<Net> &a, const std::shared_ptr<Net> & b) {
            if (a->getSinkSize() != b->getSinkSize()) return a->getSinkSize() < b->getSinkSize();
            auto guide_a = a->getGuide();
            auto guide_b = b->getGuide();
            return (guide_a.end_x - guide_a.start_x) * (guide_a.end_y - guide_a.start_y) < (guide_b.end_x - guide_b.start_x) * (guide_b.end_y - guide_b.start_y) ;
        };
        std::sort(netlist.begin(), netlist.end(), cmp);

        RTree rtree;

        std::vector<tf::Task> tasks;
        int totalNets = netlist.size();
        for (int i = 0; i < totalNets; i++) {
            // std::cout << i << std::endl;
            auto net = netlist[i];
            AStarPathfinder* finder = new AStarPathfinder(net, graph);
            tasks.push_back(taskflow.emplace([finder, net](){
                // std::cout << "i = " << i << ' ' << "netlist.size = "  << netlist.size() << std::endl;
                net->setRouteStatus(finder->run());
                delete finder;
            }));
            auto guide = netlist[i]->getGuide();
            boostBox box(boostPoint(guide.start_x, guide.start_y),
                         boostPoint(guide.end_x, guide.end_y));
            std::vector<std::pair<boostBox, int>> results;
            rtree.query(bgi::intersects(box), std::back_inserter(results));
            for (auto result : results) {
                tasks[i].succeed(tasks[result.second]);
            }
            rtree.insert({box, i});
            // for (int j = 0; j < i; j++) {
            //     auto guide_a = netlist[i]->getGuide();
            //     auto guide_b = netlist[j]->getGuide();
            //     if (((guide_a.start_x >= guide_b.start_x && guide_a.start_x <= guide_b.end_x)
            //         ||(guide_a.end_x >= guide_b.start_x && guide_a.end_x <= guide_b.end_x))
            //          &&((guide_a.start_y >= guide_b.start_y && guide_a.start_y <= guide_b.end_y)
            //          ||(guide_a.end_y >= guide_b.start_y && guide_a.end_y <= guide_b.end_y))) {
            //              tasks[i].succeed(tasks[j]);
            //          }
            // }
        }
        tf::Executor executor(40);
        executor.run(taskflow).wait();
        return;
    }

} // namespace router
