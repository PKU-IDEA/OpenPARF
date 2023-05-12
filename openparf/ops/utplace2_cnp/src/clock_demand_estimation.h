#pragma once

// C++ standard library headers
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

// project headers
#include "container/vector_2d.hpp"
#include "geometry/geometry.hpp"
#include "util/arg.h"

namespace clock_network_planner {
using openparf::container::Vector2D;
using openparf::geometry::Box;

class ClockDemandEstimationImpl {
  public:
  using IndexType    = std::int32_t;
  using IndexSet     = std::set<IndexType>;
  using IndexBox     = Box<IndexType>;
  using Index2BoxMap = std::map<IndexType, IndexBox>;

  private:
  CLASS_ARG(Vector2D<IndexSet>, ck_demand_grid);
  CLASS_ARG(Index2BoxMap, ck2cr_box_map);
};

class ClockDemandEstimation {
  SHARED_CLASS_ARG(ClockDemandEstimation, impl);

  public:
  FORWARDED_METHOD(ck_demand_grid)
  FORWARDED_METHOD(ck2cr_box_map)
};

MAKE_SHARED_CLASS(ClockDemandEstimation)

}   // namespace clock_network_planner