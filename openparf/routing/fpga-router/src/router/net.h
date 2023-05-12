#ifndef NET_H
#define NET_H

#include "database/pin.h"
#include "utils/utils.h"

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

class Net {
public:
    Net();
    Net(std::string _name) : name(_name), routestatus(UNROUTED), _useGlobalResult(false) {
        guide.start_x = std::numeric_limits<INDEX_T>::max();
        guide.start_y = std::numeric_limits<INDEX_T>::max();
        guide.end_x = std::numeric_limits<INDEX_T>::min();
        guide.end_y = std::numeric_limits<INDEX_T>::min();
        rerouteTime = 0;
    }

    std::string getName() { return name; }

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

    void useGlobalResult(bool flag) { _useGlobalResult = flag; }
    bool useGlobalResult() { return _useGlobalResult; }
    std::set<std::pair<INDEX_T, INDEX_T>>& getGlobalRouteResult() { return globalRouteResult; }
    std::vector<int>& getSinks() { return sinks; }
    // std::shared_ptr<RouteTree> getRouteTree() {return localRouteTree;}


    void addGlobalRouteResult(INDEX_T posX, INDEX_T posY) {
        globalRouteResult.insert(std::make_pair(posX, posY)); 
        addGuideNode(posX, posY);
        }

    COST_T getNetCritial();

    COST_T getSinkCritical(int sink) { return sinkCritical[sink]; }
    void setSinkCritical(int idx, COST_T crit) { sinkCritical[sinks[idx]] = crit; }

    friend class Pathfinder;
    friend class AStarPathfinder;
    friend class RouteGraphBuilder;
    friend class BoundingBoxBuilder;
    friend class Router;
    friend class GlobalRouter;
    friend class GlobalRouteTree;
private:
    std::string name;
    int source;
    int rerouteTime;
    std::vector<int> sinks;
    std::unordered_map<int, COST_T> sinkCritical;
    std::unordered_map<int, int> sinkSet;
    int netId;
    int inQueueCnt = 0;
    int outQueueCnt = 0;

    // std::shared_ptr<RouteTree> localRouteTree;
    RouteStatus routestatus;
    BoundingBox guide;
    std::set<std::pair<INDEX_T, INDEX_T>> globalRouteResult;
    bool _useGlobalResult;
};

} // namespace router
#endif // NET_H