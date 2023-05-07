/**
 * File              : stable_div_cuda_kernel.cu
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.29.2020
 * Last Modified Date: 04.29.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeStableDiv(T const* x, T const* y, int32_t n, T* out) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    auto const& xx = x[i];
    auto const& yy = y[i];
    auto& oo = out[i];
    if (yy == 0) {
      oo = 0;
    } else {
      oo = xx / yy;
    }
  }
}
template <typename T>
void computeStableDivCudaLauncher(T const* x, T const* y, int32_t n, T* out) {
  computeStableDiv<<<ceilDiv(n, 256), 256>>>(x, y, n, out);
}

template <typename T>
__global__ void computeStableDivGrad(T const* grad_outs, T const* x, T const* y,
                                     int32_t n, T* gradx, T* grady) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    auto const& xx = x[i];
    auto const& yy = y[i];
    auto const& go = grad_outs[i];
    auto& gx = gradx[i];
    auto& gy = grady[i];
    if (yy == 0) {
      gx = 0;
      gy = 0;
    } else {
      gx = go / yy;
      gy = -xx * gx * gx * go;
    }
  }
}

template <typename T>
void computeStableDivGradCudaLauncher(T const* grad_outs, T const* x,
                                      T const* y, int32_t n, T* gradx,
                                      T* grady) {
  computeStableDivGrad<<<ceilDiv(n, 256), 256>>>(grad_outs, x, y, n, gradx,
                                                 grady);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeStableDivCudaLauncher<T>(T const *x, T const *y, int32_t n, T *out);      \
    template void computeStableDivGradCudaLauncher<T>(T const *grad_outs, T const *x, T const *y,  \
                                                      int32_t n, T *gradx, T *grady);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
