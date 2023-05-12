#include "net.h"
#include "routetree.h"

#include <limits>
#include <math.h>
namespace router {
Net::Net() {
    // localRouteTree = std::shared_ptr<RouteTree>(new RouteTree());
    guide.start_x = std::numeric_limits<INDEX_T>::max();
    guide.start_y = std::numeric_limits<INDEX_T>::max();
    guide.end_x = std::numeric_limits<INDEX_T>::min();
    guide.end_y = std::numeric_limits<INDEX_T>::min();

    routestatus = UNROUTED;
}
void Net::setSource(int _source) {
    source = _source;
}

void Net::addSink(int _sink) {
    sinks.push_back(_sink);
    sinkSet[_sink] = sinks.size() - 1;
    sinkCritical[_sink] = 0.f;
}

int Net::getSource() {
    return source;
}

int Net::getSinkByIdx(int idx) {
    return sinks[idx];
}

bool Net::isSink(int pin) {
    return (sinkSet.find(pin) != sinkSet.end());
}

void Net::addGuideNode(INDEX_T x, INDEX_T y) {
    // if (name == "net_33534") {
    //     std::cerr << "x: " << x << " y: " << y << " guide: (" << guide.start_x << ',' << guide.start_y <<")(" << guide.end_x << ',' << guide.end_y << ")" << std::endl; 
    // }
    guide.start_x = std::min(x, guide.start_x);
    guide.end_x   = std::max(x, guide.end_x);
    guide.start_y = std::min(y, guide.start_y);
    guide.end_y   = std::max(y, guide.end_y);
}

void Net::expandGuide() {
    guide.start_x = std::max(0, guide.start_x - 1);
    guide.end_x++;
    guide.start_y = std::max(0, guide.start_y - 1);
    guide.end_y++;
}

COST_T Net::getNetCritial() {
    return log((guide.start_x - guide.end_x + 1) * (guide.start_y - guide.end_y + 1) + 1) * (rerouteTime + 1);
}


}