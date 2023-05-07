/**
 * @file   pin_utilization_map.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin, Yibai Meng
 * @date   Dec 2019
 */

#include "ops/pin_utilization/src/pin_utilization_map_kernel.hpp"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// fill the demand map pin by pin
/// @param pos array of (x, y) pairs
/// @param inst_sizes array of (width, height) pairs
/// @param pin_weights array of number of pins per inst
/// @param range [begin, end) range pair of inst indices
template<typename T>
void       pinDemandMapLauncher(T const  *pos,
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
              int32_t                     deterministic_flag,
              int32_t                     num_threads,
              T                          *pin_utilization_map);

at::Tensor pinUtilizationMapForward(at::Tensor pos,
        at::Tensor                             inst_sizes,
        at::Tensor                             pin_weights,
        double                                 xl,
        double                                 yl,
        double                                 xh,
        double                                 yh,
        double                                 bin_size_x,
        double                                 bin_size_y,
        int32_t                                num_bins_x,
        int32_t                                num_bins_y,
        std::pair<int32_t, int32_t>            range,
        int32_t                                deterministic_flag) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CPU(inst_sizes);
  CHECK_EVEN(inst_sizes);
  CHECK_CONTIGUOUS(inst_sizes);

  CHECK_FLAT_CPU(pin_weights);
  CHECK_CONTIGUOUS(pin_weights);

  at::Tensor pin_utilization_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "pinDemandMapLauncher", [&] {
    pinDemandMapLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(inst_sizes, scalar_t), OPENPARF_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh,
            yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y, range, deterministic_flag, at::get_num_threads(),
            OPENPARF_TENSOR_DATA_PTR(pin_utilization_map, scalar_t));
  });

  return pin_utilization_map;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::pinUtilizationMapForward, "compute pin utilization map");
}
