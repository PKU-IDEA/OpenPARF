#pragma once
#include "util/arg.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace clock_network_planner {

class ClockInfoGroupImpl : public std::enable_shared_from_this<ClockInfoGroupImpl> {
public:
    using IndexType   = std::int32_t;
    using FlowIntType = std::int64_t;
    using IndexVector = std::vector<IndexType>;
private:
    CLASS_ARG(IndexVector, ck_sets);
    CLASS_ARG(FlowIntType, flow_supply);
};

class ClockInfoGroup {
    SHARED_CLASS_ARG(ClockInfoGroup, impl);
public:
    FORWARDED_METHOD(ck_sets)
    FORWARDED_METHOD(flow_supply)
};

MAKE_SHARED_CLASS(ClockInfoGroup)

using ClockInfoGroupArray = std::vector<ClockInfoGroup>;
using ClockInfoGroupArrayMap =
        std::unordered_map<ClockInfoGroupImpl::IndexType, ClockInfoGroupArray>;

class ClockInfoGroupArrayWrapperImpl
    : public std::enable_shared_from_this<ClockInfoGroupArrayWrapperImpl> {
    CLASS_ARG(ClockInfoGroupArray, packed_clock_info_array);
    CLASS_ARG(ClockInfoGroupArrayMap, single_ele_clock_info_array_map);
};

class ClockInfoGroupArrayWrapper {
private:
    std::shared_ptr<ClockInfoGroupArrayWrapperImpl> impl_;

public:
    explicit ClockInfoGroupArrayWrapper(std::shared_ptr<ClockInfoGroupArrayWrapperImpl> impl)
        : impl_(std::move(impl)) {}
    FORWARDED_METHOD(packed_clock_info_array)
    FORWARDED_METHOD(single_ele_clock_info_array_map)
};

MAKE_SHARED_CLASS(ClockInfoGroupArrayWrapper)

}   // namespace clock_network_planner
