
/**
 * File              : soft_floor_cuda_kernel.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__device__ T sigmoid(T x, T const *sigmoid_gamma) {
  return 1.0 / (1.0 + exp(-(*sigmoid_gamma) * x));
}

template <typename T>
__device__ T softFloor(T x, T const *soft_floor_gamma, int32_t num_floor) {
  T res = 0.0;

  for (int i = 1; i < num_floor; ++i) {
    res += sigmoid<T>(x - static_cast<T>(i), soft_floor_gamma);
  }

  return res;
}

template <typename T>
__device__ T sigmoid_grad(T x, T const *sigmoid_gamma) {
  T sig = sigmoid<T>(x, sigmoid_gamma);
  return (*sigmoid_gamma) * sig * (1.0 - sig);
}

template <typename T>
__device__ T softFloor_grad(T x, T const *soft_floor_gamma, int32_t num_floor) {
  T res = 0.0;

  for (int32_t i = 1; i < num_floor; ++i) {
    res += sigmoid_grad<T>(x - static_cast<T>(i), soft_floor_gamma);
  }

  return res;
}

// Main function to compute soft floor and its gradient.
// Processes each pin position and applies the soft floor transformation.
template <typename T>
__global__ void computeSoftFloor(T const *pos, int32_t num_pins, T xl, T yl,
                                 T slr_width, T slr_height, int32_t num_slrX,
                                 int32_t num_slrY, T const *soft_floor_gamma,
                                 T *pin_pos, T *grad_intermediate) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pins) return;
  int32_t pin_id_offset = i << 1;

  T scaled_pos_x = (pos[pin_id_offset] - xl) / slr_width;
  pin_pos[pin_id_offset] =
      softFloor<T>(scaled_pos_x, soft_floor_gamma, num_slrX);
  grad_intermediate[pin_id_offset] =
      softFloor_grad<T>(scaled_pos_x, soft_floor_gamma, num_slrX) /
      slr_width;

  T scaled_pos_y = (pos[pin_id_offset + 1] - yl) / slr_height;
  pin_pos[pin_id_offset + 1] =
      softFloor<T>(scaled_pos_y, soft_floor_gamma, num_slrY);
  grad_intermediate[pin_id_offset + 1] =
      softFloor_grad<T>(scaled_pos_y, soft_floor_gamma, num_slrY) /
      slr_height;
}

// Wrapper function to launch the soft floor computation.
template <typename T>
void computeSoftFloorCudaLauncher(T const *pos, int32_t num_pins, T xl, T yl,
                                  T slr_width, T slr_height,
                                  int32_t num_slrX, int32_t num_slrY,
                                  T const *soft_floor_gamma, T *pin_pos,
                                  T *grad_intermediate) {
  computeSoftFloor<T><<<ceilDiv(num_pins, 256), 256>>>(
      pos, num_pins, xl, yl, slr_width, slr_height, num_slrX, num_slrY,
      soft_floor_gamma, pin_pos, grad_intermediate);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                           \
  template void computeSoftFloorCudaLauncher<T>(                              \
      T const *pos, int32_t num_pins, T xl, T yl, T slr_width, T slr_height, \
      int32_t num_slrX, int32_t num_slrY, T const *soft_floor_gamma,      \
      T *pin_pos, T *grad_intermediate);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
