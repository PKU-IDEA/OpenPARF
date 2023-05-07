#pragma once

#include "util/arg.h"
#include "util/enums.h"
#include "util/message.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace clock_network_planner {

using namespace openparf;

class AssignmentGroupImpl : public std::enable_shared_from_this<AssignmentGroupImpl> {
public:
    using IndexType   = int32_t;
    using FlowIntType = int64_t;
    using RealType    = float;
    using IndexVector = std::vector<IndexType>;

    IndexType inst_num() { return inst_ids_.size(); }

private:
    CLASS_ARG(IndexVector,
              ck_sets);   ///< Clock indexes set in this assignment group sorted in
                          ///< non-decreasing order.
    CLASS_ARG(IndexVector, inst_ids);      ///< Instance indexes in this assignment group
    CLASS_ARG(RealType, x_grav);           ///< The x-coordinate of the gravity center
    CLASS_ARG(RealType, y_grav);           ///< The y-coordinate of the gravity center
    CLASS_ARG(RealType, area_dem);         ///< Total area demand of all nodes in this group
    CLASS_ARG(IndexType, pin_nums);        ///< The total pin number in this assignment group
    CLASS_ARG(FlowIntType, flow_supply);   ///< Flow supply in the min-cost-max-flow
};

class AssignmentGroup {
private:
    std::shared_ptr<AssignmentGroupImpl> impl_;

public:
    explicit AssignmentGroup(std::shared_ptr<AssignmentGroupImpl> impl) : impl_(std::move(impl)) {}

    decltype(auto) impl() { return impl_; }

    FORWARDED_METHOD(ck_sets)
    FORWARDED_METHOD(inst_ids)
    FORWARDED_METHOD(x_grav)
    FORWARDED_METHOD(y_grav)
    FORWARDED_METHOD(pin_nums)
    FORWARDED_METHOD(area_dem)
    FORWARDED_METHOD(inst_num)
    FORWARDED_METHOD(flow_supply)
};

MAKE_SHARED_CLASS(AssignmentGroup)

using AssignmentGroupArray = std::vector<AssignmentGroup>;
using AssignmentGroupArrayMap =
        std::unordered_map<AssignmentGroupImpl::IndexType, AssignmentGroupArray>;

/**
 * @brief There are one packed group array and a series of single-element group arrays.
 * 1. As for resource types like LUT & FF, instances containing these serious will be packed
 * spatially into a packed group, which make up this packed group array. All the groups in this
 * array are assigned to `one` specified target site.
 * 2. As for each SSSR(Single-Site Single-Resource) resource types like DSP and RAM, we have a
 * single-element group array for each of the SSIR resource type. `single-element group` means
 * that each group contains exactly one instance. A series of single-element groups make up a
 * single-element group array, and all the groups in a array are assigned to `one` specified target
 * site.
 * 3. Ensure that each instance can not appear in more than one group.
 */
class GroupArrayWrapperImpl : public std::enable_shared_from_this<GroupArrayWrapperImpl> {
public:
    using IndexType = AssignmentGroupImpl::IndexType;

private:
    CLASS_ARG(AssignmentGroupArray, packed_group_array);
    CLASS_ARG(AssignmentGroupArrayMap, single_ele_group_array_map);
};

class GroupArrayWrapper {
private:
    std::shared_ptr<GroupArrayWrapperImpl> impl_;

public:
    explicit GroupArrayWrapper(std::shared_ptr<GroupArrayWrapperImpl> impl)
        : impl_(std::move(impl)) {}

    FORWARDED_METHOD(packed_group_array)
    FORWARDED_METHOD(single_ele_group_array_map)
};

MAKE_SHARED_CLASS(GroupArrayWrapper)

}   // namespace clock_network_planner
