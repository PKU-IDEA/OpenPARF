#include "clock_info_group_factory.h"
#include <map>

namespace clock_network_planner {

void ClockInfoGroupFactory::genClockInfoGroupFromAssignmentGroupArray(
        AssignmentGroupArray &assignment_groups, ClockInfoGroupArray &clock_info_array) {
    using IndexVector = ClockInfoGroupImpl::IndexVector;
    using FlowIntType = ClockInfoGroupImpl::FlowIntType;
    std::map<IndexVector, FlowIntType> ck_id2flow_map;
    for (auto &group : assignment_groups) {
        ck_id2flow_map[group.ck_sets()] += group.flow_supply();
    }
    clock_info_array.clear();
    for (auto &iter : ck_id2flow_map) {
        auto &ck_id_sets  = iter.first;
        auto &flow_supply = iter.second;
        clock_info_array.push_back(detail::makeClockInfoGroup());
        clock_info_array.back().ck_sets()     = ck_id_sets;
        clock_info_array.back().flow_supply() = flow_supply;
    }
}


ClockInfoGroupArrayWrapper ClockInfoGroupFactory::genClockInfoGroupWrapperFromGroupArrayWrapper(
        GroupArrayWrapper group_array_wrapper) {
    auto clock_info_group_array_wrapper = detail::makeClockInfoGroupArrayWrapper();
    genClockInfoGroupFromAssignmentGroupArray(
            group_array_wrapper.packed_group_array(),
            clock_info_group_array_wrapper.packed_clock_info_array());
    for (auto &iter : group_array_wrapper.single_ele_group_array_map()) {
        genClockInfoGroupFromAssignmentGroupArray(
                iter.second,
                clock_info_group_array_wrapper.single_ele_clock_info_array_map()[iter.first]);
    }
    return clock_info_group_array_wrapper;
}
}   // namespace clock_network_planner