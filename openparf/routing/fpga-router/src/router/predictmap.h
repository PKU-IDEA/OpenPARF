#ifndef PREDICT_MAP_H
#define PREDICT_MAP_H

#include "database/builder_template.h"
#include "routegraph.h"

#include <memory>
#include <unordered_map>
namespace router {
    class PredictMap {
    public:
        PredictMap() {}
        PredictMap(std::shared_ptr<RouteGraph> _graph) : graph(_graph) {}

        void initPredictMap(std::shared_ptr<database::GridLayout> layout);
        COST_T predictDist(int source, int sink);
    
    private:
        std::shared_ptr<RouteGraph> graph;
        
        std::unordered_map<std::shared_ptr<database::Pin>, int> pinIdx;
        std::vector<std::vector<COST_T> > sameGridDist;
        std::vector<std::vector<COST_T> > diffGridDist;
    };
}


#endif //PREDICT_MAP_H