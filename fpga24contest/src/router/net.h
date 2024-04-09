#ifndef NET_H
#define NET_H

#include "database/pin.h"
#include "utils/utils.h"
#include "thirdparty/taskflow/taskflow.hpp"

#include <memory>
#include <vector>
#include <set>
#include <limits>
#include <unordered_map>
#include <unordered_set>
namespace router {

enum RouteStatus {
    SUCCESS,
    UNROUTED,
    FAILED,
    CONGESTED
};


class RouteTree;
class GlobalSubRouteTree;

class Net {
public:
    Net();
    Net(std::string _name) : name(_name), routestatus(UNROUTED) {
        guide.start_x = std::numeric_limits<INDEX_T>::max();
        guide.start_y = std::numeric_limits<INDEX_T>::max();
        guide.end_x = std::numeric_limits<INDEX_T>::min();
        guide.end_y = std::numeric_limits<INDEX_T>::min();
        rerouteTime = 0;
    }

    std::string getName() { return name; }
    int getId() { return netId; }

    void setSource(int _source);
    void addSink(int _sink);

    int getSource();
    int getSinkByIdx(int idx);
    int getSinkSize() { return sinks.size(); }
    bool isSink(int pinIdx);
    int getSinkIdx(int sink) {
        if (!isSink(sink)) return -1;
        return sinkSet[sink];
    } 
    RouteStatus getRouteStatus() { return routestatus; }
    void setRouteStatus(RouteStatus status) { routestatus = status; }
    BoundingBox& getGuide() { return guide; }

    void addGuideNode(INDEX_T x, INDEX_T y);
    void expandGuide();
    void addRerouteTime() { ++rerouteTime; }
    int getRerouteTime() { return rerouteTime; }

    std::vector<int>& getSinks() { return sinks; }
    // std::shared_ptr<RouteTree> getRouteTree() {return localRouteTree;}

    void clearCongestedVertices() { congestedVeticesNum = 0; }
    void addCongestedVertices() { congestedVeticesNum++; }
    int getCongestedVertices() { return congestedVeticesNum; }

    void setNetWL(COST_T wl) { netWL = wl; }
    COST_T getNetWL() { return netWL; }

    void setDurationTimeMs(COST_T duration) { durationTimeMs = duration; }
    COST_T getDurationTimeMs() { return durationTimeMs; }

    void setIndirect(bool v = true) { indirect = v; }
    bool isIndirect() { return indirect; }

    friend class Pathfinder;
    friend class AStarPathfinder;
    friend class Router;
private:
    std::string name;
    int source;
    int rerouteTime;
    std::vector<int> sinks;
    std::unordered_map<int, int> sinkSet;
    int netId;
    int inQueueCnt = 0;
    int outQueueCnt = 0;

    int inQueueCntPin = 0;
    int inQueueCntChildren = 0;

    // std::shared_ptr<RouteTree> localRouteTree;
    RouteStatus routestatus;
    BoundingBox guide;

    int congestedVeticesNum;

    int erasedNodes;
    int netWL;
    double durationTimeMs;
    bool indirect = false;

};

} // namespace router
#endif // NET_H