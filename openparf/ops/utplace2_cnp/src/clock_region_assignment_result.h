#pragma once

#include "util/arg.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace clock_network_planner {

class ClockRegionAssignmentResultWrapperImpl {
public:
    using IndexType          = std::int32_t;
    using IndexVector        = std::vector<IndexType>;
    using CrToInstIdArray    = std::vector<IndexVector>;
    using CrToInstIdArrayMap = std::unordered_map<IndexType, CrToInstIdArray>;

private:
    CLASS_ARG(CrToInstIdArray, packed_inst_array);
    CLASS_ARG(CrToInstIdArrayMap, single_ele_inst_map);
    CLASS_ARG(CrToInstIdArray, cr_to_all_inst_array);
};

class ClockRegionAssignmentResultWrapper {
    SHARED_CLASS_ARG(ClockRegionAssignmentResultWrapper, impl);

public:
    FORWARDED_METHOD(packed_inst_array);
    FORWARDED_METHOD(single_ele_inst_map);
    FORWARDED_METHOD(cr_to_all_inst_array); ///< combine the results of packed instances and SSIR instances.
};

MAKE_SHARED_CLASS(ClockRegionAssignmentResultWrapper)

}   // namespace clock_network_planner
