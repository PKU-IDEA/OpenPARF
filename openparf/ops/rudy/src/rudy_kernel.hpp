/**
 * File              : rudy_kernel.hpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.10.2020
 * Last Modified Date: 07.15.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_RUDY_SRC_RUDY_KERNEL_HPP_
#define OPENPARF_OPS_RUDY_SRC_RUDY_KERNEL_HPP_

#include <algorithm>
#include <limits>

#include "ops/rudy/src/risa_parameters.h"
#include "util/atomic_ops.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template<typename T>
inline DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

template<typename T, typename V, typename AtomicOp>
void rudyKernel(const T*         pin_pos,
        const V*                 netpin_start,
        const V*                 flat_netpin,
        const T*                 net_weights,
        const T                  bin_size_x,
        const T                  bin_size_y,
        const T                  xl,
        const T                  yl,
        const T                  xh,
        const T                  yh,
        const V                  num_bins_x,
        const V                  num_bins_y,
        const V                  num_nets,
        AtomicOp                 atomic_op,
        const V                  num_threads,
        typename AtomicOp::type* horizontal_utilization_map,
        typename AtomicOp::type* vertical_utilization_map) {
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;
  int32_t chunk_size     = OPENPARF_STD_NAMESPACE::max(int32_t(num_nets / num_threads / 16), 1);

#pragma omp parallel for num_threads(num_threads) schedule(static, chunk_size)
  for (int32_t i = 0; i < (int32_t) num_nets; i++) {
    T x_max = std::numeric_limits<T>::lowest();
    T x_min = std::numeric_limits<T>::max();
    T y_max = std::numeric_limits<T>::lowest();
    T y_min = std::numeric_limits<T>::max();

    for (auto j = netpin_start[i]; j < netpin_start[i + 1]; j++) {
      auto    pin_id = flat_netpin[j];
      // |pin_pos| is in the form of xyxyxy...
      const T xx     = pin_pos[pin_id << 1];
      const T yy     = pin_pos[(pin_id << 1) + 1];
      x_max          = OPENPARF_STD_NAMESPACE::max(x_max, xx);
      x_min          = OPENPARF_STD_NAMESPACE::min(x_min, xx);
      y_max          = OPENPARF_STD_NAMESPACE::max(y_max, yy);
      y_min          = OPENPARF_STD_NAMESPACE::min(y_min, yy);
    }

    // compute the bin box that this net will affect
    auto bin_index_xl = int32_t((x_min - xl) * inv_bin_size_x);
    auto bin_index_xh = int32_t((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl      = OPENPARF_STD_NAMESPACE::max(bin_index_xl, (decltype(bin_index_xl)) 0);
    bin_index_xh      = OPENPARF_STD_NAMESPACE::min(bin_index_xh, (decltype(bin_index_xh)) num_bins_x);
    auto bin_index_yl = int32_t((y_min - yl) * inv_bin_size_y);
    auto bin_index_yh = int32_t((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl      = OPENPARF_STD_NAMESPACE::max(bin_index_yl, (decltype(bin_index_yl)) 0);
    bin_index_yh      = OPENPARF_STD_NAMESPACE::min(bin_index_yh, (decltype(bin_index_yh)) num_bins_y);


    T wt              = netWiringDistributionMapWeight<T>(netpin_start[i + 1] - netpin_start[i]);
    if (net_weights) {
      wt *= net_weights[i];
    }

    for (auto x = bin_index_xl; x < bin_index_xh; x++) {
      for (auto y = bin_index_yl; y < bin_index_yh; y++) {
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
        overlap *= wt;
        auto index = x * num_bins_y + y;
        /**
         * Follow Wuxi's implementation, a tolerance is added to avoid
         * 0-size bounding box
         */
        atomic_op(&horizontal_utilization_map[index], overlap / (y_max - y_min + std::numeric_limits<T>::epsilon()));
        atomic_op(&vertical_utilization_map[index], overlap / (x_max - x_min + std::numeric_limits<T>::epsilon()));
      }
    }
  }
}

template<typename T, typename V>
void rudyLauncher(const T* pin_pos,
        const V*           netpin_start,
        const V*           flat_netpin,
        const T*           net_weights,
        const T            bin_size_x,
        const T            bin_size_y,
        const T            xl,
        const T            yl,
        const T            xh,
        const T            yh,
        const V            num_bins_x,
        const V            num_bins_y,
        const V            num_nets,
        int32_t            deterministic_flag,
        int32_t            num_threads,
        T*                 horizontal_utilization_map,
        T*                 vertical_utilization_map) {
  if (deterministic_flag) {
    using AtomicIntType                   = int64_t;
    AtomicIntType            scale_factor = 1e10;
    AtomicAdd<AtomicIntType> atomic_op(scale_factor);

    AtomicIntType*           buf_hmap = new AtomicIntType[num_bins_x * num_bins_y];
    AtomicIntType*           buf_vmap = new AtomicIntType[num_bins_x * num_bins_y];
    DEFER({
      delete[] buf_hmap;
      delete[] buf_vmap;
    });

    copyScaleArray(buf_hmap, horizontal_utilization_map, scale_factor, num_bins_x * num_bins_y, num_threads);
    copyScaleArray(buf_vmap, vertical_utilization_map, scale_factor, num_bins_x * num_bins_y, num_threads);

    rudyKernel<T, V, decltype(atomic_op)>(pin_pos, netpin_start, flat_netpin, net_weights, bin_size_x, bin_size_y, xl,
            yl, xh, yh, num_bins_x, num_bins_y, num_nets, atomic_op, num_threads, buf_hmap, buf_vmap);

    copyScaleArray(horizontal_utilization_map, buf_hmap, static_cast<T>(1.0 / scale_factor), num_bins_x * num_bins_y,
            num_threads);
    copyScaleArray(vertical_utilization_map, buf_vmap, static_cast<T>(1.0 / scale_factor), num_bins_x * num_bins_y,
            num_threads);

  } else {
    AtomicAdd<T> atomic_op;
    rudyKernel<T, V, decltype(atomic_op)>(pin_pos, netpin_start, flat_netpin, net_weights, bin_size_x, bin_size_y, xl,
            yl, xh, yh, num_bins_x, num_bins_y, num_nets, atomic_op, num_threads, horizontal_utilization_map,
            vertical_utilization_map);
  }
}
OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_RUDY_SRC_RUDY_KERNEL_HPP_
