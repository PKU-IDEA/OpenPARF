/**
 * File              : rudy_kernel.hpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.10.2020
 * Last Modified Date: 07.15.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_RUDY_KERNEL_HPP
#define OPENPARF_RUDY_KERNEL_HPP

#include "risa_parameters.h"
#include "util/util.h"


OPENPARF_BEGIN_NAMESPACE

template <typename T>
inline DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

template <typename T, typename V>
int32_t rudyLauncher(
    const T* pin_pos,
    const V* netpin_start,//每个net的起始pin位置
    const V* flat_netpin,//展平后的pinid
    const T *net_weights,
    const T bin_size_x,
    const T bin_size_y,
    const T xl,
    const T yl,
    const T xh,
    const T yh,
    const V num_bins_x,
    const V num_bins_y,
    const V num_nets,
    const V num_threads,
    const T* pinDirects,
    T* horizontal_utilization_map,
    T* vertical_utilization_map,
    T* pin_density_map){

  const T inv_bin_size_x = 1.0  / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;


  int32_t chunk_size = OPENPARF_STD_NAMESPACE::max(int32_t(num_nets / num_threads / 16), 1);
  #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for(int32_t i = 0; i < (int32_t)num_nets; i++){
    T x_max = std::numeric_limits<T>::lowest();
    T x_min = std::numeric_limits<T>::max();
    T y_max = std::numeric_limits<T>::lowest();
    T y_min = std::numeric_limits<T>::max();

    auto pin_out_x=int32_t(0);
    auto pin_out_y=int32_t(0);

    for(auto j = netpin_start[i]; j < netpin_start[i + 1]; j++){//net连接pin的数量
      auto pin_id = flat_netpin[j];
      const T xx = pin_pos[pin_id<<1];
      const T yy = pin_pos[(pin_id<<1) + 1];
      x_max = OPENPARF_STD_NAMESPACE::max(x_max, xx);
      x_min = OPENPARF_STD_NAMESPACE::min(x_min, xx);
      y_max = OPENPARF_STD_NAMESPACE::max(y_max, yy);
      y_min = OPENPARF_STD_NAMESPACE::min(y_min, yy);

      //pin density
      auto x_index = int32_t((xx - xl) * inv_bin_size_x);
      auto y_index = int32_t((yy - yl) * inv_bin_size_y);
      auto xy_index = x_index * num_bins_y + y_index;
      pin_density_map[xy_index] += 1;

      if(pinDirects[pin_id] == 1)
      {
        pin_out_x = x_index;
        pin_out_y = y_index;
      }
    }

    // compute the bin box that this net will affect
    auto bin_index_xl = int32_t((x_min - xl) * inv_bin_size_x);
    auto bin_index_xh = int32_t((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl = OPENPARF_STD_NAMESPACE::max(bin_index_xl, (decltype(bin_index_xl))0);
    bin_index_xh = OPENPARF_STD_NAMESPACE::min(bin_index_xh, (decltype(bin_index_xh))num_bins_x);
    auto bin_index_yl = int32_t((y_min - yl) * inv_bin_size_y);
    auto bin_index_yh = int32_t((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl = OPENPARF_STD_NAMESPACE::max(bin_index_yl, (decltype(bin_index_yl))0);
    bin_index_yh = OPENPARF_STD_NAMESPACE::min(bin_index_yh, (decltype(bin_index_yh))num_bins_y);

    T wt = netWiringDistributionMapWeight<T>(netpin_start[i + 1] - netpin_start[i]);

    if (net_weights){
      wt *= net_weights[i];
    }


    for(auto x = bin_index_xl; x < bin_index_xh; x++){
      for(auto y = bin_index_yl; y < bin_index_yh; y++){
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap = 1;
        auto index = x * num_bins_y + y;
        if ((x == pin_out_x) || (y == pin_out_y)){
          overlap *= 2;
        }
        /**
         * Follow Wuxi's implementation, a tolerance is added to avoid
         * 0-size bounding box
         */
        #pragma omp atomic update
        horizontal_utilization_map[index] +=
            overlap / (y_max - y_min + std::numeric_limits<T>::epsilon());
        #pragma omp atomic update
        vertical_utilization_map[index] +=
            overlap / (x_max - x_min + std::numeric_limits<T>::epsilon());
      }
    }
  }


  return 0;
}


OPENPARF_END_NAMESPACE
#endif // OPENPARF_RUDY_CPU_KERNEL_H
