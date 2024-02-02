/**
 * File              : sll_cuda.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeSLLCudaLauncher(T const *pos, int32_t const *flat_netpin,
                            int32_t const *netpin_start,
                            uint8_t const *net_mask, int32_t *sll_counts_table,
                            int32_t *partial_sll, int32_t num_nets, T xl, T yl,
                            T slr_width, T slr_height, int32_t num_slrX, int32_t num_slrY);

/// @brief Compute super long line counts
/// @param pos pin locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or not
/// @param xl lower bound coordinate in the x-dimension.
/// @param yl lower bound coordinate in the y-dimension.
/// @param slr_width width of the SLR
/// @param slr_height height of the SLR
/// @param num_slrX number of columns in the SLR topology
/// @param num_slrY number of rows in the SLR topology
at::Tensor sllForward(at::Tensor pos, at::Tensor flat_netpin,
                      at::Tensor netpin_start, at::Tensor net_weights,
                      at::Tensor net_mask, at::Tensor sll_counts_table,
                      double xl, double yl, double slr_width,
                      double slr_height, int32_t num_slrX, int32_t num_slrY) {
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
  CHECK_FLAT_CUDA(sll_counts_table);
  CHECK_CONTIGUOUS(sll_counts_table);

  int32_t    num_nets    = net_mask.numel();
  at::Tensor partial_sll = at::zeros({num_nets, 1}, flat_netpin.options());

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeSLLCudaLauncher", [&] {
    computeSLLCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(flat_netpin, int32_t),
        OPENPARF_TENSOR_DATA_PTR(netpin_start, int32_t),
        OPENPARF_TENSOR_DATA_PTR(net_mask, uint8_t),
        OPENPARF_TENSOR_DATA_PTR(sll_counts_table, int32_t),
        OPENPARF_TENSOR_DATA_PTR(partial_sll, int32_t), num_nets, xl, yl,
        slr_width, slr_height, num_slrX, num_slrY);
  });

  return partial_sll;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::sllForward,
        "super long line counts forward computation (CUDA)");
}