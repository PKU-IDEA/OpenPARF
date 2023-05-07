/**
 * File              : hpwl.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.21.2020
 * Last Modified Date: 04.21.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeHPWLLauncher(T const *pos, int32_t const *flat_netpin,
                         int32_t const *netpin_start, uint8_t const *net_mask,
                         int32_t num_nets, int32_t num_threads,
                         T *partial_hpwl);

/// @brief Compute half-perimeter wirelength
/// @param pos cell locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
at::Tensor hpwlForward(at::Tensor pos, at::Tensor flat_netpin,
                       at::Tensor netpin_start, at::Tensor net_weights,
                       at::Tensor net_mask) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  int32_t num_nets = netpin_start.numel() - 1;
  at::Tensor partial_hpwl = at::zeros({num_nets, 2}, pos.options());
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeHPWLLauncher", [&] {
    computeHPWLLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(flat_netpin, int32_t),
        OPENPARF_TENSOR_DATA_PTR(netpin_start, int32_t),
        OPENPARF_TENSOR_DATA_PTR(net_mask, uint8_t), num_nets,
        at::get_num_threads(),
        OPENPARF_TENSOR_DATA_PTR(partial_hpwl, scalar_t));
  });
  if (net_weights.numel()) {
    partial_hpwl.mul_(net_weights.view({num_nets, 1}));
  }
  return partial_hpwl;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::hpwlForward,
        "Half-perimeter wirelength forward");
}
