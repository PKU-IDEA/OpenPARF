/**
 * @file   pin_utilization_map_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin, Yibai Meng
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94 Copied from DREAMPlace
 */

#include <cuda_runtime.h>

#include "util/atomic_ops.cuh"
#include "util/limits.cuh"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template<typename T>
__device__ __forceinline__ void clamp(T &x, T min_x, T max_x) {
  if (x < min_x) {
    x = min_x;
  } else if (x > max_x) {
    x = max_x;
  }
}

// fill the demand map net by net
/// pin distributions are smoothed within the instance bounding box.
template<typename T, typename AtomicOp>
__global__ void pinDemandMapKernel(T const *pos,
        T const                            *inst_sizes,
        T const                            *pin_weights,
        T                                   xl,
        T                                   yl,
        T                                   xh,
        T                                   yh,
        T                                   bin_size_x,
        T                                   bin_size_y,
        int32_t                             num_bins_x,
        int32_t                             num_bins_y,
        thrust::pair<int32_t, int32_t>      range,
        int32_t                             num_threads,
        typename AtomicOp::type            *pin_utilization_map,
        AtomicOp                            atomic_add) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x + range.first;

  if (i < range.second) {
    const T inv_bin_size_x = 1.0 / bin_size_x;
    const T inv_bin_size_y = 1.0 / bin_size_y;

    int32_t offset         = i << 1;
    const T inst_center_x  = pos[offset];
    const T inst_center_y  = pos[offset | 1];
    const T inst_size_x    = inst_sizes[offset];
    const T inst_size_y    = inst_sizes[offset | 1];

    // ignore the cell whose area is zero
    if (inst_size_x < cuda::numeric_limits<T>::epsilon() || inst_size_y < cuda::numeric_limits<T>::epsilon()) {
      return;
    }

    T x_min = inst_center_x - inst_size_x * 0.5;
    T x_max = inst_center_x + inst_size_x * 0.5;
    T y_min = inst_center_y - inst_size_y * 0.5;
    T y_max = inst_center_y + inst_size_y * 0.5;
    clamp(x_min, xl, xh);
    clamp(x_max, xl, xh);
    clamp(y_min, yl, yh);
    clamp(y_max, yl, yh);

    int32_t bin_index_xl = int32_t((x_min - xl) * inv_bin_size_x);
    int32_t bin_index_xh = int32_t((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl         = OPENPARF_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh         = OPENPARF_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int32_t bin_index_yl = int32_t((y_min - yl) * inv_bin_size_y);
    int32_t bin_index_yh = int32_t((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl         = OPENPARF_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh         = OPENPARF_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T area               = inst_size_x * inst_size_y;
    T density            = pin_weights[i] / area;
    for (int32_t x = bin_index_xl; x < bin_index_xh; ++x) {
      for (int32_t y = bin_index_yl; y < bin_index_yh; ++y) {
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap =
                OPENPARF_STD_NAMESPACE::max(
                        OPENPARF_STD_NAMESPACE::min(x_max, bin_xh) - OPENPARF_STD_NAMESPACE::max(x_min, bin_xl),
                        (T) 0) *
                OPENPARF_STD_NAMESPACE::max(
                        OPENPARF_STD_NAMESPACE::min(y_max, bin_yh) - OPENPARF_STD_NAMESPACE::max(y_min, bin_yl), (T) 0);
        int32_t index = x * num_bins_y + y;
        atomic_add(&pin_utilization_map[index], overlap * density);
      }
    }
  }
}

template<typename T, typename V>
__global__ void fixedPoint2FloatingPoint(T *target, V const *src, int32_t size, T scale_factor) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    target[i] = src[i] * scale_factor;
  }
}

// fill the demand map net by net
template<typename T>
void pinDemandMapCudaLauncher(T const *pos,
        T const                       *inst_sizes,
        T const                       *pin_weights,
        T                              xl,
        T                              yl,
        T                              xh,
        T                              yh,
        T                              bin_size_x,
        T                              bin_size_y,
        int32_t                        num_bins_x,
        int32_t                        num_bins_y,
        std::pair<int32_t, int32_t>    range,
        int32_t                        deterministic_flag,
        int32_t                        num_threads,
        T                             *pin_utilization_map) {
  if (deterministic_flag) {
    using AtomicIntType                   = unsigned long long int;
    AtomicIntType            scale_factor = 1e10;
    AtomicAdd<AtomicIntType> atomic_add(scale_factor);
    AtomicIntType           *pin_utilization_map_fixed = nullptr;
    int32_t                  num_bins                  = num_bins_x * num_bins_y;
    allocateCUDA(pin_utilization_map_fixed, num_bins);
    DEFER({ destroyCUDA(pin_utilization_map_fixed); });

    cudaMemset(pin_utilization_map_fixed, 0, sizeof(AtomicIntType) * num_bins);
    pinDemandMapKernel<<<ceilDiv(range.second - range.first, 512), 512>>>(pos, inst_sizes, pin_weights, xl, yl, xh, yh,
            bin_size_x, bin_size_y, num_bins_x, num_bins_y, thrust::make_pair(range.first, range.second), num_threads,
            pin_utilization_map_fixed, atomic_add);

    fixedPoint2FloatingPoint<<<ceilDiv(num_bins, 512), 512>>>(pin_utilization_map, pin_utilization_map_fixed, num_bins,
            (T) 1.0 / atomic_add.scaleFactor());
  } else {
    AtomicAdd<T> atomic_add;
    int32_t      num_bins = num_bins_x * num_bins_y;
    cudaMemset(pin_utilization_map, 0, sizeof(T) * num_bins);
    pinDemandMapKernel<<<ceilDiv(range.second - range.first, 512), 512>>>(pos, inst_sizes, pin_weights, xl, yl, xh, yh,
            bin_size_x, bin_size_y, num_bins_x, num_bins_y, thrust::make_pair(range.first, range.second), num_threads,
            pin_utilization_map, atomic_add);
  }
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void pinDemandMapCudaLauncher<T>(T const *pos, T const *inst_sizes, const T *pin_weights, T xl, T yl, T xh, \
          T yh, T bin_size_x, T bin_size_y, int32_t num_bins_x, int32_t num_bins_y, std::pair<int32_t, int32_t> range, \
          int32_t deterministic_flag, int32_t num_threads, T *pin_utilization_map);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

OPENPARF_END_NAMESPACE
