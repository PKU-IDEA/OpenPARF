#ifndef SCHEDULER_H
#define SCHEDULER_H
#include <vector>
#include <memory>

#include "net.h"
#include "pathfinder.h"
#include "pathfinderastar.h"
#include "routegraph.h"
#include "utils/rtree.h"


namespace router
{
    class Scheduler {
    public:
        Scheduler() {}
        Scheduler(std::shared_ptr<RouteGraph> _graph) : graph(_graph) {}
        std::vector<std::vector<int> >& schedule(std::vector<std::shared_ptr<Net>>& netlist);
        void taskflowSchedule(std::vector<std::shared_ptr<Net>>& netlist);

    private:
        int layoutWidth;
        int layoutHeight;

        // RTree rtree;

        std::shared_ptr<RouteGraph> graph;
        std::vector<std::vector<int> > batches;
    };
} // namespace router

#endif // SCHEDULER_H