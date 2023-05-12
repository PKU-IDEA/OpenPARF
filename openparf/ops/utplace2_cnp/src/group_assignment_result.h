#pragma once
#include "util/arg.h"
#include "util/enums.h"
#include "util/message.h"
#include <memory>
#include <unordered_map>

namespace clock_network_planner {

using namespace openparf;

struct GroupToRegionAssignmentResult {
    using IndexType   = std::int32_t;
    using FlowIntType = std::int64_t;

    CLASS_ARG(IndexType, cr_id);
    CLASS_ARG(FlowIntType, flow_size);

public:
    GroupToRegionAssignmentResult(IndexType cr_id, FlowIntType flow_size)
        : cr_id_(cr_id), flow_size_(flow_size) {}
};

using GroupAssignmentResultArray = std::vector<std::vector<GroupToRegionAssignmentResult>>;
using GroupAssignmentResultArrayMap =
        std::unordered_map<GroupToRegionAssignmentResult::IndexType, GroupAssignmentResultArray>;

class AssignmentResultWrapperImpl : public std::enable_shared_from_this<AssignmentResultWrapperImpl> {
    CLASS_ARG(GroupAssignmentResultArray, packed_group_array);
    CLASS_ARG(GroupAssignmentResultArrayMap, single_ele_group_array_map);
};

class AssignmentResultWrapper {
    SHARED_CLASS_ARG(AssignmentResultWrapper, impl);

public:
    FORWARDED_METHOD(packed_group_array);
    FORWARDED_METHOD(single_ele_group_array_map);
};

MAKE_SHARED_CLASS(AssignmentResultWrapper)
}   // namespace clock_network_planner
