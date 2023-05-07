#pragma once

#include "util/arg.h"
#include <set>
#include <vector>
#include <memory>

namespace clock_network_planner {

    class CrId2CkSinkSet {
    public:
        using IndexType = std::int32_t;
        using IndexSet = std::set<IndexType>;
        using InstanceType = std::vector<IndexSet>;
    private:
        std::shared_ptr<InstanceType> impl_;
    public:
        explicit CrId2CkSinkSet(std::shared_ptr<InstanceType> impl) : impl_(std::move(impl)) {}
        FORWARDED_METHOD(clear)
        FORWARDED_METHOD(resize)
        FORWARDED_METHOD(at)
    };
    CUSTOM_MAKE_SHARED_CLASS(CrId2CkSinkSet, CrId2CkSinkSet::InstanceType)
}


