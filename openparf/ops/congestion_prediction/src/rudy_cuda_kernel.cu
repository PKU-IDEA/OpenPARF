/**
 * File              : rudy_cuda_kernel.cu
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.10.2020
 * Last Modified Date: 07.10.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */

#include "cuda_runtime.h"
#include "risa_parameters.h"
#include "util/limits.cuh"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
inline __device__ DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

template <typename T>
__global__ void
RudyCudaKernel(T *pin_pos, int32_t *netpin_start, int32_t *flat_netpin,
               T *net_weights, T bin_size_x, T bin_size_y, T xl, T yl, T xh,
               T yh, int32_t num_bins_x, int32_t num_bins_y, int32_t num_nets,
               T *pinDirects, T *horizontal_utilization_map,
               T *vertical_utilization_map, T *pin_density_map) {

  int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  T inv_bin_size_x = 1.0  / bin_size_x;
  T inv_bin_size_y = 1.0 / bin_size_y;
  if (i < num_nets) {
    int32_t start = netpin_start[i];
    int32_t end = netpin_start[i + 1];

    T x_max = -cuda::numeric_limits<T>::max();
    T x_min = cuda::numeric_limits<T>::max();
    T y_max = -cuda::numeric_limits<T>::max();
    T y_min = cuda::numeric_limits<T>::max();

    int32_t pin_out_x=int32_t(0);
    int32_t pin_out_y=int32_t(0);
    for (int32_t j = start; j < end; ++j) {
      int32_t pin_id = flat_netpin[j];
      T xx = pin_pos[pin_id << 1];
      T yy = pin_pos[(pin_id << 1) | 1];
      x_max = OPENPARF_STD_NAMESPACE::max(xx, x_max);
      x_min = OPENPARF_STD_NAMESPACE::min(xx, x_min);
      y_max = OPENPARF_STD_NAMESPACE::max(yy, y_max);
      y_min = OPENPARF_STD_NAMESPACE::min(yy, y_min);

      //pin_density
      int32_t x_index = int32_t((xx - xl) * inv_bin_size_x);
      int32_t y_index = int32_t((yy - yl) * inv_bin_size_y);
      int32_t xy_index = x_index * num_bins_y + y_index;
      atomicAdd(pin_density_map + xy_index, 1);

      if(pinDirects[pin_id] == 1)
      {
        pin_out_x = x_index;
        pin_out_y = y_index;
      }
    }

    int32_t bin_index_xl = int32_t((x_min - xl) / bin_size_x);
    int32_t bin_index_xh = int32_t((x_max - xl) / bin_size_x) + 1;
    bin_index_xl = OPENPARF_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = OPENPARF_STD_NAMESPACE::min(bin_index_xh, num_bins_x);
    int32_t bin_index_yl = int32_t((y_min - yl) / bin_size_y);
    int32_t bin_index_yh = int32_t((y_max - yl) / bin_size_y) + 1;
    bin_index_yl = OPENPARF_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = OPENPARF_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T wt = netWiringDistributionMapWeight<T>(end - start);
    if (net_weights) {
      wt *= net_weights[i];
    }
    for (int32_t x = bin_index_xl; x < bin_index_xh; ++x) {
      for (int32_t y = bin_index_yl; y < bin_index_yh; ++y) {
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap = 1;
        int32_t index = x * num_bins_y + y;
        if ((x == pin_out_x) || (y == pin_out_y)){
          overlap *= 2;
        }
        /**
         * Follow Wuxi's implementation, a tolerance is added to avoid
         * 0-size bounding box
         */
        atomicAdd(horizontal_utilization_map + index,
                  overlap /
                      (y_max - y_min + cuda::numeric_limits<T>::epsilon()));
        atomicAdd(vertical_utilization_map + index,
                  overlap /
                      (x_max - x_min + cuda::numeric_limits<T>::epsilon()));
      }
    }
  }
}

template <typename T>
int32_t
RudyCudaLauncher(T *pin_pos, int32_t *netpin_start, int32_t *flat_netpin,
                 T *net_weights, T bin_size_x, T bin_size_y, T xl, T yl, T xh,
                 T yh, int32_t num_bins_x, int32_t num_bins_y, int32_t num_nets,
                 T *pinDirects, T *horizontal_utilization_map,
                 T *vertical_utilization_map, T *pin_density_map) {
  int32_t thread_count = 256;
  int32_t block_count = ceilDiv(num_nets, thread_count);
  RudyCudaKernel<<<(uint32_t)block_count, {(uint32_t)thread_count, 1u, 1u}>>>(
      pin_pos, netpin_start, flat_netpin, net_weights, bin_size_x, bin_size_y,
      xl, yl, xh, yh, num_bins_x, num_bins_y, num_nets, pinDirects, horizontal_utilization_map,
      vertical_utilization_map, pin_density_map);
  return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template int32_t RudyCudaLauncher<T>(                                                          \
            T * pin_pos, int32_t * netpin_start, int32_t * flat_netpin, T * net_weights,           \
            T bin_size_x, T bin_size_y, T xl, T yl, T xh, T yh, int32_t num_bins_x,                \
            int32_t num_bins_y, int32_t num_nets, T * pinDirects,                                  \
            T * horizontal_utilization_map, T * vertical_utilization_map, T * pin_density_map);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE