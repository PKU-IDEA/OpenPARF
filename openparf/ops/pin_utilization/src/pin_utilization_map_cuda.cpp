/**
 * @file   pin_utilization_map_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin, Yibai Meng
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94 Copied from DREAMPlace
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// fill the demand map pin by pin
/// @param pos array of (x, y) pairs
/// @param inst_sizes array of (width, height) pairs
/// @param pin_weights array of number of pins per inst
/// @param range array of [begin, end) range pairs of inst indices
template<typename T>
void       pinDemandMapCudaLauncher(T const *pos,
              T const                       *inst_sizes,
              T const                       *pin_weights,
              T                              xl,
              T                              yl,
              T                              xh,
              T                              yh,
              T                              bin_size_x,
              T                              bin_size_y,
              int32_t                        num_bins_x,
              int32_t                        num_bins_y,
              std::pair<int32_t, int32_t>    range,
              int32_t                        deterministic_flag,
              int32_t                        num_threads,
              T                             *pin_utilization_map);

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
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CUDA(inst_sizes);
  CHECK_EVEN(inst_sizes);
  CHECK_CONTIGUOUS(inst_sizes);

  CHECK_FLAT_CUDA(pin_weights);
  CHECK_CONTIGUOUS(pin_weights);

  at::Tensor pin_utilization_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "pinDemandMapCudaLauncher", [&] {
    pinDemandMapCudaLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(inst_sizes, scalar_t), OPENPARF_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh,
            yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y, range, deterministic_flag, at::get_num_threads(),
            OPENPARF_TENSOR_DATA_PTR(pin_utilization_map, scalar_t));
  });

  return pin_utilization_map;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::pinUtilizationMapForward, "compute pin utilization map (CUDA)");
}
