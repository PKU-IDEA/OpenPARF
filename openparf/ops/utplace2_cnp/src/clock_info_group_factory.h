#include "assignment_group.h"
#include "clock_info_group.h"

namespace clock_network_planner {

struct ClockInfoGroupFactory {
    static void genClockInfoGroupFromAssignmentGroupArray(AssignmentGroupArray &assignment_groups,
                                                          ClockInfoGroupArray & clock_info_array);

    static ClockInfoGroupArrayWrapper
    genClockInfoGroupWrapperFromGroupArrayWrapper(GroupArrayWrapper group_array_wrapper);
};
}   // namespace clock_network_planner
