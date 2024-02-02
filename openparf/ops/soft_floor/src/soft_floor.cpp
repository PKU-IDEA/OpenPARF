/**
 * File              : soft_floor.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

// Function template declaration for computeSoftFloorLauncher
template <typename T>
void computeSoftFloorLauncher(T const* pos, int32_t num_pins, T xl, T yl,
                              T slr_width, T slr_height, int32_t num_slrX,
                              int32_t num_slrY, T const* soft_floor_gamma,
                              int32_t num_threads, T* pin_pos, T* scaled_pos);

/// @brief Compute the soft floor transformation for pin locations.
/// @param pos Tensor of pin locations in x and y directions.
/// @param xl lower bound coordinate in the x-dimension.
/// @param yl lower bound coordinate in the y-dimension.
/// @param slr_width Width of the SLR.
/// @param slr_height Height of the SLR.
/// @param num_slrX Number of columns in the SLR topology
/// @param num_slrY Number of rows in the SLR topology
/// @param soft_floor_gamma Gamma parameter for the soft floor calculation.
/// @return A vector of two tensors - the output positions and intermediate gradients.
std::vector<at::Tensor> softFloorForward(at::Tensor pos, double xl, double yl,
                                         double slr_width, double slr_height,
                                         int32_t num_slrX, int32_t num_slrY,
                                         at::Tensor soft_floor_gamma) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  auto    out               = at::zeros_like(pos);
  auto    grad_intermediate = at::zeros_like(pos);
  int32_t num_pins          = pos.numel() / 2;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeSoftFloorLauncher", [&] {
    computeSoftFloorLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(pos, scalar_t), num_pins, xl, yl, slr_width,
        slr_height, num_slrX, num_slrY,
        OPENPARF_TENSOR_DATA_PTR(soft_floor_gamma, scalar_t),
        at::get_num_threads(), OPENPARF_TENSOR_DATA_PTR(out, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(grad_intermediate, scalar_t));
  });

  return {out, grad_intermediate};
}

// Backward pass for the SoftFloor operation.
at::Tensor softFloorBackward(at::Tensor grad_pos,
                             at::Tensor grad_intermediate) {
  CHECK_FLAT_CPU(grad_pos);
  CHECK_EVEN(grad_pos);
  CHECK_CONTIGUOUS(grad_pos);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  return grad_out;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::softFloorForward, "softFloor forward");
  m.def("backward", &OPENPARF_NAMESPACE::softFloorBackward,
        "softFloor backward");
}
