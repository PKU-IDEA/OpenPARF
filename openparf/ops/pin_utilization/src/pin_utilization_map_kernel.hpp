/**
 * File              : pin_utilization_map_kernel.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.14.2020
 * Last Modified Date: 07.15.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_PIN_UTILIZATION_SRC_PIN_UTILIZATION_MAP_KERNEL_HPP_
#define OPENPARF_OPS_PIN_UTILIZATION_SRC_PIN_UTILIZATION_MAP_KERNEL_HPP_

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "util/atomic_ops.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template<typename T>
inline void clamp(T &x, T min_x, T max_x) {
  if (x < min_x) {
    x = min_x;
  } else if (x > max_x) {
    x = max_x;
  }
}

/// @brief fill the demand map pin by pin
/// pin distributions are smoothed within the instance bounding box.
template<typename T, typename AtomicOp>
void pinDemandMapKernel(T const    *pos,
        T const                    *inst_sizes,
        T const                    *pin_weights,
        T                           xl,
        T                           yl,
        T                           xh,
        T                           yh,
        T                           bin_size_x,
        T                           bin_size_y,
        int32_t                     num_bins_x,
        int32_t                     num_bins_y,
        std::pair<int32_t, int32_t> range,
        int32_t                     num_threads,
        typename AtomicOp::type    *pin_utilization_map,
        AtomicOp                    atomic_add) {
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int32_t chunk_size     = OPENPARF_STD_NAMESPACE::max(int32_t((range.second - range.first) / num_threads / 16), 1);
  // #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int32_t i = range.first; i < range.second; ++i) {
    int32_t offset        = i << 1;
    const T inst_center_x = pos[offset];
    const T inst_center_y = pos[offset | 1];
    const T inst_size_x   = inst_sizes[offset];
    const T inst_size_y   = inst_sizes[offset | 1];

    // ignore the cell whose area is zero
    if (inst_size_x < std::numeric_limits<T>::epsilon() || inst_size_y < std::numeric_limits<T>::epsilon()) {
      continue;
    }

    T x_min = inst_center_x - inst_size_x * 0.5;
    T x_max = inst_center_x + inst_size_x * 0.5;
    T y_min = inst_center_y - inst_size_y * 0.5;
    T y_max = inst_center_y + inst_size_y * 0.5;
    clamp(x_min, xl, xh);
    clamp(x_max, xl, xh);
    clamp(y_min, yl, yh);
    clamp(y_max, yl, yh);

    int32_t bin_index_xl = int32_t((x_min - xl) * inv_bin_size_x);
    int32_t bin_index_xh = int32_t((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl         = OPENPARF_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh         = OPENPARF_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int32_t bin_index_yl = int32_t((y_min - yl) * inv_bin_size_y);
    int32_t bin_index_yh = int32_t((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl         = OPENPARF_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh         = OPENPARF_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T area               = inst_size_x * inst_size_y;
    T density            = pin_weights[i] / area;
    for (int32_t x = bin_index_xl; x < bin_index_xh; ++x) {
      for (int32_t y = bin_index_yl; y < bin_index_yh; ++y) {
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap =
                OPENPARF_STD_NAMESPACE::max(
                        OPENPARF_STD_NAMESPACE::min(x_max, bin_xh) - OPENPARF_STD_NAMESPACE::max(x_min, bin_xl),
                        (T) 0) *
                OPENPARF_STD_NAMESPACE::max(
                        OPENPARF_STD_NAMESPACE::min(y_max, bin_yh) - OPENPARF_STD_NAMESPACE::max(y_min, bin_yl), (T) 0);
        int32_t index = x * num_bins_y + y;
        atomic_add(&pin_utilization_map[index], overlap * density);
      }
    }
  }
}

template<typename T, typename V>
void        fixedPoint2FloatingPoint(T *target, V const *src, int32_t size, T scale_factor, int32_t num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < size; ++i) {
    target[i] = src[i] * scale_factor;
  }
}

template<typename T>
void OPENPARF_NOINLINE pinDemandMapLauncher(T const *pos,
        T const                                     *inst_sizes,
        T const                                     *pin_weights,
        T                                            xl,
        T                                            yl,
        T                                            xh,
        T                                            yh,
        T                                            bin_size_x,
        T                                            bin_size_y,
        int32_t                                      num_bins_x,
        int32_t                                      num_bins_y,
        std::pair<int32_t, int32_t>                  range,
        int32_t                                      deterministic_flag,
        int32_t                                      num_threads,
        T                                           *pin_utilization_map) {
  if (deterministic_flag) {
    using AtomicIntType                     = int64_t;
    AtomicIntType              scale_factor = 1e10;
    AtomicAdd<AtomicIntType>   atomic_add(scale_factor);
    std::vector<AtomicIntType> pin_utilization_map_fixed(num_bins_x * num_bins_y, 0);
    pinDemandMapKernel(pos, inst_sizes, pin_weights, xl, yl, xh, yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y,
            range, num_threads, pin_utilization_map_fixed.data(), atomic_add);
    fixedPoint2FloatingPoint(pin_utilization_map, pin_utilization_map_fixed.data(), pin_utilization_map_fixed.size(),
            (T) 1.0 / atomic_add.scaleFactor(), num_threads);

  } else {
    AtomicAdd<T> atomic_add;
    int32_t      num_bins = num_bins_x * num_bins_y;
    memset(pin_utilization_map, 0, num_bins * sizeof(T));
    pinDemandMapKernel(pos, inst_sizes, pin_weights, xl, yl, xh, yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y,
            range, num_threads, pin_utilization_map, atomic_add);
  }
}

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_PIN_UTILIZATION_SRC_PIN_UTILIZATION_MAP_KERNEL_HPP_
