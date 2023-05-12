/**
 * @file   move_boundary_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Apr 2020
 */

#include "cuda_runtime.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeMoveBoundary(T* pos, T const* node_sizes, T xl, T yl,
                                    T xh, T yh,
                                    thrust::pair<int32_t, int32_t> range) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t i = tid + range.first;
  if (range.first < range.second) {
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

template<typename T>
void computeMoveBoundaryCudaLauncher(T *pos, T const *node_sizes, T xl, T yl, T xh, T yh,
                                     std::pair<int32_t, int32_t> const &range) {
    computeMoveBoundary<<<ceilDiv(range.second - range.first, 256), 256>>>(
            pos, node_sizes, xl, yl, xh, yh, thrust::make_pair(range.first, range.second));
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeMoveBoundaryCudaLauncher<T>(T * pos, T const *node_sizes, T xl, T yl,     \
                                                     T xh, T yh,                                   \
                                                     std::pair<int32_t, int32_t> const &range);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
