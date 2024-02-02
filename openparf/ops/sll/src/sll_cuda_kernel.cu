/**
 * File              : sll_cuda_kernel.cu
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeSLLCounts(T const *pos, int32_t const *flat_netpin,
                                 int32_t const *netpin_start, uint8_t const *net_mask,
                                 int32_t *sll_counts_table, int32_t *partial_sll,
                                 int32_t num_nets, T xl, T yl, T slr_width, T slr_height,
                                 int32_t num_slrX, int32_t num_slrY) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_nets) return;

  auto                   &sll                = partial_sll[i];
  constexpr const int32_t net_ignore_pin_num = 2000;
  const int32_t           num_slrs           = num_slrX * num_slrY;
  int32_t                 pin_num            = netpin_start[i + 1] - netpin_start[i];
  if (pin_num > net_ignore_pin_num) return;

  if (!net_mask[i]) return;

  int32_t slr_dist[4]    = {0};  // Initialize to 0
  int32_t temp           = 0;
  T       inv_slr_width  = 1.0 / slr_width;
  T       inv_slr_height = 1.0 / slr_height;

  for (int32_t j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
    int32_t offset               = flat_netpin[j] << 1;
    int32_t xx                   = static_cast<int32_t>((pos[offset] - xl) * inv_slr_width);
    int32_t yy                   = static_cast<int32_t>((pos[offset + 1] - yl) * inv_slr_height);
    slr_dist[xx * num_slrY + yy] = 1;
  }
  for (int32_t idx = 0; idx < num_slrs; idx++) {
    temp |= slr_dist[idx] << (num_slrs - idx - 1);
  }
  sll = sll_counts_table[temp];
}

template <typename T>
void computeSLLCudaLauncher(T const *pos, int32_t const *flat_netpin,
                            int32_t const *netpin_start, uint8_t const *net_mask,
                            int32_t *sll_counts_table, int32_t *partial_sll,
                            int32_t num_nets, T xl, T yl, T slr_width, T slr_height,
                            int32_t num_slrX, int32_t num_slrY) {
  computeSLLCounts<T><<<ceilDiv(num_nets, 256), 256, 1>>>(
      pos, flat_netpin, netpin_start, net_mask, sll_counts_table, partial_sll,
      num_nets, xl, yl, slr_width, slr_height, num_slrX, num_slrY);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                          \
  template void computeSLLCudaLauncher<T>(                                   \
      T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start, \
      uint8_t const *net_mask, int32_t *sll_counts_table,                    \
      int32_t *partial_sll, int32_t num_nets, T xl, T yl, T slr_width,       \
      T slr_height, int32_t num_slrX, int32_t num_slrY);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE