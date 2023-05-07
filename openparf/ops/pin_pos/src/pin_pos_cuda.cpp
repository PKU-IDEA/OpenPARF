/**
 * File              : pin_pos_cuda.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.21.2020
 * Last Modified Date: 04.21.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Given cell locations, compute pin locations
/// @param x cell locations in x direction
/// @param y cell locations in y direction
/// @param pin_offset_x pin offset in x direction
/// @param pin_offset_y pin offset in y direction
/// @param pin2node_map map pin index to node index
/// @param node_pins map node index to pins
/// @param node_pins_start start index of node_pins for each
/// node
/// @param num_pins number of pins
/// @param pin_x pin positions in x direction
/// @param pin_y pin positions in y direction
template <typename T>
void computePinPosCudaLauncher(T const* pos, T const* pin_offsets,
                               int32_t const* pin2node_map, int32_t num_pins,
                               T* pin_pos);

template <typename T>
void computePinPosGradCudaLauncher(T const* grad_outs, int32_t const* node_pins,
                                   int32_t const* node_pins_start,
                                   int32_t num_nodes, T* grads);

/// @brief Given cell locations, compute pin locations
/// @param pos cell locations in x and then y direction
/// @param pin_offset_x pin offset in x direction
/// @param pin_offset_y pin offset in y direction
/// @param pin2node_map map pin index to node index
/// @param node_pins map node index to pins
/// @param node_pins_start start index of node_pins for each
/// node
/// @param num_nodes number of nodes
/// @param num_pins number of pins
/// @return pin positions in x and then y direction
at::Tensor pinPosForward(at::Tensor pos, at::Tensor pin_offsets,
                         at::Tensor pin2node_map) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  auto out = at::zeros_like(pin_offsets);
  int32_t num_pins = pin2node_map.numel();

  OPENPARF_DISPATCH_FLOATING_TYPES(
      pos, "computePinPosCudaLauncher", [&] {
        computePinPosCudaLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(pin_offsets, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(pin2node_map, int32_t), num_pins,
            OPENPARF_TENSOR_DATA_PTR(out, scalar_t));
      });

  return out;
}

at::Tensor pinPosBackward(at::Tensor grad_out, at::Tensor pos,
                          at::Tensor node_pins, at::Tensor node_pins_start) {
  CHECK_FLAT_CUDA(grad_out);
  CHECK_EVEN(grad_out);
  CHECK_CONTIGUOUS(grad_out);

  int32_t num_physical_nodes = node_pins_start.numel() - 1;
  // this number of nodes includes fillers as well,
  // which is larger than node_pins_start
  auto out = at::zeros_like(pos);

  OPENPARF_DISPATCH_FLOATING_TYPES(
      grad_out, "computePinPosGradCudaLauncher", [&] {
        computePinPosGradCudaLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(grad_out, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(node_pins, int32_t),
            OPENPARF_TENSOR_DATA_PTR(node_pins_start, int32_t),
            num_physical_nodes, OPENPARF_TENSOR_DATA_PTR(out, scalar_t));
      });

  return out;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::pinPosForward, "PinPos forward (CUDA)");
  m.def("backward", &OPENPARF_NAMESPACE::pinPosBackward,
        "PinPos backward (CUDA)");
}
