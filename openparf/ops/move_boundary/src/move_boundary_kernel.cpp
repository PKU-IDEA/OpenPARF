/**
 * @file   move_boundary_kernel.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeMoveBoundary(T* pos, T const* node_sizes, T xl, T yl, T xh, T yh,
                         std::pair<int32_t, int32_t> const& range,
                         int32_t num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = range.first; i < range.second; ++i) {
    auto offset = (i << 1);
    auto& cx = pos[offset];
    auto& cy = pos[offset + 1];
    auto const& w = node_sizes[offset];
    auto const& h = node_sizes[offset + 1];

    cx = OPENPARF_STD_NAMESPACE::max(cx, xl + w / 2);
    cx = OPENPARF_STD_NAMESPACE::min(cx, xh - w / 2);

    cy = OPENPARF_STD_NAMESPACE::max(cy, yl + h / 2);
    cy = OPENPARF_STD_NAMESPACE::min(cy, yh - h / 2);
  }
}

template <typename T>
void OPENPARF_NOINLINE computeMoveBoundaryLauncher(
    T* pos, T const* node_sizes, T xl, T yl, T xh, T yh,
    std::pair<int32_t, int32_t> const& range, int32_t num_threads) {
  computeMoveBoundary(pos, node_sizes, xl, yl, xh, yh, range, num_threads);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                     \
  void instantiateComputeMoveBoundaryLauncher(                          \
      T* pos, T const* node_sizes, T xl, T yl, T xh, T yh,              \
      std::pair<int32_t, int32_t> const& range, int32_t num_threads) {  \
    computeMoveBoundaryLauncher(pos, node_sizes, xl, yl, xh, yh, range, \
                                num_threads);                           \
  }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
