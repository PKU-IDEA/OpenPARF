/**
 * @file   hpwl_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Compute half-perimeter wirelength
 */

#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWL(T const *pos, int32_t const *flat_netpin,
                            int32_t const *netpin_start,
                            uint8_t const *net_mask, int32_t num_nets,
                            T *partial_hpwl) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_nets) {
    T max_x = -FLT_MAX;
    T min_x = FLT_MAX;

    auto &hpwl = partial_hpwl[(i << 1) + threadIdx.y]; // x and y
    constexpr const int32_t net_ignore_pin_num = 2000;
    int32_t pin_num = netpin_start[i + 1] - netpin_start[i];
    // ignore extra large nets
    // their hpwl tend to not change that much while taking up
    // most of GPU's compute time
    // This heuristic is not needed on CPU
    if(pin_num > net_ignore_pin_num) {
      hpwl = 0;
      return;
    }
    if (net_mask[i]) {
      for (int32_t j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        int32_t offset = (flat_netpin[j] << 1) + threadIdx.y;
        min_x = min(min_x, pos[offset]);
        max_x = max(max_x, pos[offset]);
      }
      hpwl = max_x - min_x;
    } else {
      hpwl = 0;
    }
  }
}

template <typename T>
void computeHPWLCudaLauncher(T const *      pos,   ///< {x, y} pairs
                             int32_t const *flat_netpin, int32_t const *netpin_start,
                             uint8_t const *net_mask, int32_t num_nets,
                             T *partial_hpwl   ///< length of num_nets * 2
) {
    computeHPWL<<<ceilDiv(num_nets, 256), {256, 2, 1}>>>(pos, flat_netpin, netpin_start, net_mask,
                                                         num_nets, partial_hpwl);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeHPWLCudaLauncher<T>(T const *pos, int32_t const *flat_netpin,             \
                                             int32_t const *netpin_start, uint8_t const *net_mask, \
                                             int32_t num_nets, T *partial_hpwl);
REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
