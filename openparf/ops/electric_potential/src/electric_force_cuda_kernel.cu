/**
 * @file   electric_force_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Aug 2018
 */

#include "cuda_runtime.h"
#include "ops/density_map/src/density_function.h"
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
__global__ void computeElectricForce(
    T const *grad_pos, T const *pos, T const *node_sizes, T const *node_weights,
    T **field_map_xs, T **field_map_ys,
    int32_t *bin_map_dims, double xl, double yl, double xh, double yh,
    thrust::pair<int32_t, int32_t> range, int32_t num_area_types, T *grad_out) {
  auto width = xh - xl;
  auto height = yh - yl;

  auto func = triangleDensity<T>;

  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t i = tid + range.first;
  if (i < range.second) {
    int32_t offset = (i << 1);
    auto node_cx = pos[offset];
    auto node_cy = pos[offset + 1];
    auto &gx = grad_out[offset];
    auto &gy = grad_out[offset + 1];
    gx = 0;
    gy = 0;
    for (int32_t area_type = 0; area_type < num_area_types; ++area_type) {
      offset = i * num_area_types * 2 + area_type * 2;
      auto node_w = node_sizes[offset];
      auto node_h = node_sizes[offset + 1];
      // skip zeros
      if (node_w * node_h == 0) {
        continue;
      }
      auto node_xl = node_cx - node_w / 2;
      auto node_yl = node_cy - node_h / 2;
      auto node_xh = node_cx + node_w / 2;
      auto node_yh = node_cy + node_h / 2;
      auto node_weight = node_weights[(offset >> 1)];

      // for each area type
      auto area_grad_pos = grad_pos[area_type];
      node_weight *= area_grad_pos;

      // get map information for a specific area type
      offset = (area_type << 1);
      auto dim_x = bin_map_dims[offset];
      auto dim_y = bin_map_dims[offset + 1];

      auto field_map_x = field_map_xs[area_type];
      auto field_map_y = field_map_ys[area_type];
      auto bin_w = width / dim_x;
      auto bin_h = height / dim_y;
      auto inv_bin_w = 1.0 / bin_w;
      auto inv_bin_h = 1.0 / bin_h;

      int32_t bin_ixl = int32_t((node_xl - xl) * inv_bin_w);
      // exclusive
      int32_t bin_ixh = int32_t(((node_xh - xl) * inv_bin_w)) + 1;
      bin_ixl = OPENPARF_STD_NAMESPACE::max(bin_ixl, 0);
      bin_ixh = OPENPARF_STD_NAMESPACE::min(bin_ixh, dim_x);

      int32_t bin_iyl = int32_t((node_yl - yl) * inv_bin_h);
      // exclusive
      int32_t bin_iyh = int32_t(((node_yh - yl) * inv_bin_h)) + 1;
      bin_iyl = OPENPARF_STD_NAMESPACE::max(bin_iyl, 0);
      bin_iyh = OPENPARF_STD_NAMESPACE::min(bin_iyh, dim_y);

      // compute gradient
      T local_gx = 0;
      T local_gy = 0;
      for (int32_t k = bin_ixl; k < bin_ixh; ++k) {
        T bin_xl = xl + k * bin_w;
        T px = func(node_xl, node_xh, bin_xl, bin_xl + bin_w);

        for (int32_t h = bin_iyl; h < bin_iyh; ++h) {
          T bin_yl = yl + h * bin_h;
          T py = func(node_yl, node_yh, bin_yl, bin_yl + bin_h);
          T area = px * py;

          int32_t idx = k * dim_y + h;
          local_gx += area * field_map_x[idx];
          local_gy += area * field_map_y[idx];
        }
      }
      local_gx *= node_weight;
      local_gy *= node_weight;
      gx += local_gx;
      gy += local_gy;
    }
  }
}

template <typename T>
void computeElectricForceCudaLauncher(
    T const *grad_pos, T const *pos, T const *node_sizes, T const *node_weights,
    std::vector<T *> &field_map_x_ptrs,
    std::vector<T *> &field_map_y_ptrs, int32_t *bin_map_dims, double xl,
    double yl, double xh, double yh, std::pair<int32_t, int32_t> const &range,
    int32_t num_area_types,
    T *grad_out) {
  T **field_map_x_ptrs_d = nullptr;
  allocateCopyCUDA(field_map_x_ptrs_d, field_map_x_ptrs.data(),
                   field_map_x_ptrs.size());
  T **field_map_y_ptrs_d = nullptr;
  allocateCopyCUDA(field_map_y_ptrs_d, field_map_y_ptrs.data(),
                   field_map_y_ptrs.size());

  computeElectricForce<<<ceilDiv(range.second - range.first, 64), 64>>>(
      grad_pos, pos, node_sizes, node_weights,
      field_map_x_ptrs_d, field_map_y_ptrs_d, bin_map_dims, xl, yl, xh, yh,
      thrust::make_pair(range.first, range.second), num_area_types, grad_out);

  destroyCUDA(field_map_x_ptrs_d);
  destroyCUDA(field_map_y_ptrs_d);
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computeElectricForceCudaLauncher<T>(                                             \
            T const *grad_pos, T const *pos, T const *node_sizes, T const *node_weights,           \
            std::vector<T *> &field_map_x_ptrs, std::vector<T *> &field_map_y_ptrs,                \
            int32_t *bin_map_dims, double xl, double yl, double xh, double yh,                     \
            std::pair<int32_t, int32_t> const &range, int32_t num_area_types, T *grad_out);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
