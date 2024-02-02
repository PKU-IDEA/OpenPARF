/**
 * File              : wasll_cuda.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

// CUDA launcher for computing weighted average super long line (WASLL).
// This function computes the WASLL for a set of pins using CUDA for acceleration.
template <typename T>
void computeWASLLCudaLauncher(T const *pos, int32_t const *flat_netpin,
                              int32_t const *netpin_start,
                              uint8_t const *net_mask, int32_t num_nets,
                              int32_t num_slrX, int32_t num_slrY,
                              T const *inv_gamma, T *partial_sll,
                              T *grad_intermediate);

// CUDA launcher for integrating net weights into the gradient computation.
// This function applies the net weights to the gradients of each net.
template <typename T>
void integrateNetWeightsCudaLauncher(const int32_t *pin2net_map,
                                     const uint8_t *net_mask,
                                     const T *net_weights, T *grad_tensor,
                                     int32_t num_pins);

/// @brief Forward pass to compute weighted average super long line and its gradient.
///
/// @param pos location of pins, x array followed by y array.
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin. The length is number of nets. The last entry equals to the
/// number of pins.
/// @param net_weights weight of nets
/// @param net_mask whether compute the wirelength for a net or not
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @param num_slrX number of super logic regions along the X axis.
/// @param num_slrY number of super logic regions along the Y axis.
/// @return Total cost as a tensor.
std::vector<at::Tensor> wasllForward(at::Tensor pos, at::Tensor flat_netpin,
                                     at::Tensor netpin_start,
                                     at::Tensor pin2net_map,
                                     at::Tensor net_weights,
                                     at::Tensor net_mask, at::Tensor inv_gamma,
                                     int32_t num_slrX, int32_t num_slrY) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int32_t num_nets = netpin_start.numel() - 1;

  at::Tensor partial_wasll = at::zeros({num_nets, 2}, pos.options());
  at::Tensor grad_intermediate = at::zeros_like(pos);

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeWASLLCudaLauncher", [&] {
    computeWASLLCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(flat_netpin, int32_t),
        OPENPARF_TENSOR_DATA_PTR(netpin_start, int32_t),
        OPENPARF_TENSOR_DATA_PTR(net_mask, uint8_t), num_nets, num_slrX,
        num_slrY, OPENPARF_TENSOR_DATA_PTR(inv_gamma, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(partial_wasll, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(grad_intermediate, scalar_t));
  });

  if (net_weights.numel()) {
    partial_wasll.mul_(net_weights.view({num_nets, 1}));
  }

  auto wasll = partial_wasll.sum();
  return {wasll, grad_intermediate};
}

/// @brief Backward pass to compute gradients for the weighted-average SLL computation.
///
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor wasllBackward(at::Tensor grad_pos, at::Tensor pos,
                         at::Tensor grad_intermediate, at::Tensor flat_netpin,
                         at::Tensor netpin_start, at::Tensor pin2net_map,
                         at::Tensor net_weights, at::Tensor net_mask,
                         at::Tensor inv_gamma) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CUDA(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);
  int32_t    num_pins = pos.numel() / 2;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeWASLLCudaLauncher", [&] {
    if (net_weights.numel()) {
      integrateNetWeightsCudaLauncher(
          OPENPARF_TENSOR_DATA_PTR(pin2net_map, int32_t),
          OPENPARF_TENSOR_DATA_PTR(net_mask, uint8_t),
          OPENPARF_TENSOR_DATA_PTR(net_weights, scalar_t),
          OPENPARF_TENSOR_DATA_PTR(grad_out, scalar_t), num_pins);
    }
  });
  return grad_out;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::wasllForward,
        "Weighted-average sll forward (CUDA)");
  m.def("backward", &OPENPARF_NAMESPACE::wasllBackward,
        "Weighted-average sll backward (CUDA)");
}
