/**
 * File              : clock_availability.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.13.2021
 * Last Modified Date: 09.13.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_DATABASE_CLOCK_AVAILABILITY_H_
#define OPENPARF_DATABASE_CLOCK_AVAILABILITY_H_

// C++ standard library headers
#include <functional>
#include <utility>
#include <vector>

// project headers
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE
template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
using ClockAvailCheckerType = std::function<bool(/* instance index */ int32_t,
                                                 /* the x coordinate of target position */ const T &,
                                                 /* the y coordinate of target position */ const T &)>;

template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
using LayoutXy2GridIndexFunctorType = std::function<uint32_t(
        /* the x coordinate of the point */ T,
        /* the y coordinate of the point */ T)>;

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
using ClockIndex2RegionIndexCheckerType = std::function<bool(const T &, const T &)>;

/// @brief Generate a function that checks if a instances in allowed at a certain location
/// @param inst_to_clock_indexes #(num_inst) tensor, clock index.
/// @param clock_available_clock_region #(num_clk, cr_width, cr_height) bool tensor. Whether a clock is allowed inside a
/// clock region.
/// @param xy_to_clock_region_index finds what clock region a location is in
template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
ClockAvailCheckerType<T> GenClockAvailChecker(const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
                                              at::Tensor                               clock_available_clock_region,
                                              LayoutXy2GridIndexFunctorType<T>         xy_to_clock_region_index) {
  openparfAssert(clock_available_clock_region.dtype() == torch::kUInt8);

  auto clock_available_clock_region_a = clock_available_clock_region.accessor<uint8_t, 2>();

  return [=](int32_t instance_id, const T &x, const T &y) -> bool {
    auto &clock_indexes = inst_to_clock_indexes[instance_id];
    if (clock_indexes.empty()) {
      // The instance is not connected to any clock net/signal.
      return true;
    }
    int32_t clock_region_id = xy_to_clock_region_index(x, y);
    // Convert site coordinates to clock region grid indexes.
    for (int32_t ck_idx : clock_indexes) {
      openparfAssert(clock_available_clock_region_a[ck_idx][clock_region_id] == 0 ||
                     clock_available_clock_region_a[ck_idx][clock_region_id] == 1);
      if (clock_available_clock_region_a[ck_idx][clock_region_id] == 0) {
        // Violate the clock region constraint
        // openparfPrint(kDebug, "FAIL%i %f %f %i %i\n", instance_id, x, y, ck_idx, clock_region_id);
        // openparfAssert(false);
        return false;
      }
    }


    return true;
  };
}

template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
LayoutXy2GridIndexFunctorType<T> GenLayoutXyGridIndexFunctor(const std::pair<double, double> &  grid_size_xy,
                                                             const std::pair<int32_t, int32_t> &grid_num_xy) {
  using IndexType           = int32_t;

  double    inv_grid_size_x = 1.0 / grid_size_xy.first;
  double    inv_grid_size_y = 1.0 / grid_size_xy.second;
  IndexType grid_num_y      = grid_num_xy.second;

  return [=](const T &layout_x, const T &layout_y) -> IndexType {
    IndexType grid_idx_x = std::floor(layout_x * inv_grid_size_x);
    IndexType grid_idx_y = std::floor(layout_y * inv_grid_size_y);
    return grid_idx_x * grid_num_y + grid_idx_y;
  };
}

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
ClockIndex2RegionIndexCheckerType<T> GenClockIndex2RegionIndexCheckerType(at::Tensor clock_available_clock_region) {
  openparfAssert(clock_available_clock_region.dtype() == torch::kUInt8);

  auto clock_available_clock_region_a = clock_available_clock_region.accessor<uint8_t, 2>();
  return [=](const T &clock_idx, const T &clock_region_id) -> bool {
    return clock_available_clock_region_a[clock_idx][clock_region_id] != 0;
  };
}
OPENPARF_END_NAMESPACE

#endif   // OPENPARF_DATABASE_CLOCK_AVAILABILITY_H_
