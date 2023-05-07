/**
 * File              : stable_div_kernel.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.29.2020
 * Last Modified Date: 04.29.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeStableDiv(T const* x, T const* y, int32_t n, int32_t num_threads,
                      T* out) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < n; ++i) {
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
void OPENPARF_NOINLINE computeStableDivLauncher(T const* x, T const* y,
                                                int32_t n, int32_t num_threads,
                                                T* out) {
  computeStableDiv(x, y, n, num_threads, out);
}

template <typename T>
void computeStableDivGrad(T const* grad_outs, T const* x, T const* y, int32_t n,
                          int32_t num_threads, T* gradx, T* grady) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < n; ++i) {
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
void OPENPARF_NOINLINE computeStableDivGradLauncher(T const* grad_outs,
                                                    T const* x, T const* y,
                                                    int32_t n,
                                                    int32_t num_threads,
                                                    T* gradx, T* grady) {
  computeStableDivGrad(grad_outs, x, y, n, num_threads, gradx, grady);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeStableDivLauncher<T>(T const *x, T const *y, int32_t n,                   \
                                              int32_t num_threads, T *out);                        \
    template void computeStableDivGradLauncher(T const *grad_outs, T const *x, T const *y,         \
                                               int32_t n, int32_t num_threads, T *gradx,           \
                                               T *grady);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
