/**
 * @file   wawl_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Apr 2020
 */

#include "assert.h"
#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeWAWL(T const *pos, int32_t const *flat_netpin,
                            int32_t const *netpin_start,
                            uint8_t const *net_mask, int32_t num_nets,
                            T const *inv_gamma, T *partial_wl,
                            T *grad_intermediate) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t xy = threadIdx.y;
  if (i < num_nets && net_mask[i]) {
    T x_max = -FLT_MAX;
    T x_min = FLT_MAX;
    int32_t bgn = netpin_start[i];
    int32_t end = netpin_start[i + 1];
    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = (flat_netpin[j] << 1) + xy;
      T xx = pos[offset];
      x_max = max(xx, x_max);
      x_min = min(xx, x_min);
    }

    T xexp_x_sum = 0;
    T xexp_nx_sum = 0;
    T exp_x_sum = 0;
    T exp_nx_sum = 0;
    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = (flat_netpin[j] << 1) + xy;
      T xx = pos[offset];
      T exp_x = exp((xx - x_max) * (*inv_gamma));
      T exp_nx = exp((x_min - xx) * (*inv_gamma));

      xexp_x_sum += xx * exp_x;
      xexp_nx_sum += xx * exp_nx;
      exp_x_sum += exp_x;
      exp_nx_sum += exp_nx;
    }

    partial_wl[(i << 1) + xy] =
        xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;

    T b_x = (*inv_gamma) / (exp_x_sum);
    T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
    T b_nx = -(*inv_gamma) / (exp_nx_sum);
    T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = (flat_netpin[j] << 1) + xy;
      T xx = pos[offset];
      auto &grad = grad_intermediate[offset];

      T exp_x = exp((xx - x_max) * (*inv_gamma));
      T exp_nx = exp((x_min - xx) * (*inv_gamma));

      grad = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
    }
  }
}

template <typename T>
void computeWAWLCudaLauncher(T const *pos, int32_t const *flat_netpin,
                             int32_t const *netpin_start,
                             uint8_t const *net_mask, int32_t num_nets,
                             T const *inv_gamma, T *partial_wl,
                             T *grad_intermediate) {
  // separate x and y
  computeWAWL<<<ceilDiv(num_nets, 64), {64, 2}>>>(
      pos, flat_netpin, netpin_start, net_mask, num_nets, inv_gamma, partial_wl,
      grad_intermediate);
}

template <typename T>
__global__ void
integrateNetWeights(int32_t const *pin2net_map, uint8_t const *net_mask,
                    T const *net_weights, T *grad_tensor, int32_t num_pins) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pins) {
    int32_t net_id = pin2net_map[i];
    T weight = net_weights[net_id];
    if (net_id >= 0 && net_mask[net_id]) {
      int32_t offset = (i << 1);
      grad_tensor[offset] *= weight;
      grad_tensor[offset + 1] *= weight;
    }
  }
}

template <typename T>
void integrateNetWeightsCudaLauncher(int32_t const *pin2net_map,
                                     uint8_t const *net_mask,
                                     T const *net_weights, T *grad_tensor,
                                     int32_t num_pins) {
  integrateNetWeights<<<ceilDiv(num_pins, 256), 256>>>(
      pin2net_map, net_mask, net_weights, grad_tensor, num_pins);
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeWAWLCudaLauncher<T>(T const *pos, int32_t const *flat_netpin,             \
                                             int32_t const *netpin_start, uint8_t const *net_mask, \
                                             int32_t num_nets, T const *inv_gamma, T *partial_wl,  \
                                             T *grad_intermediate);                                \
    template void integrateNetWeightsCudaLauncher<T>(                                              \
            int32_t const *pin2net_map, uint8_t const *net_mask, T const *net_weights,             \
            T *grad_tensor, int32_t num_pins);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
