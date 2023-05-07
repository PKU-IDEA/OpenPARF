/**
 * @file   density_map_kernel.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Compute density map on CPU
 */

#include "util/atomic_ops.h"
#include "ops/density_map/src/density_function.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Perfrm a += b * scale_factor
template<typename T, typename U, typename W>
void        scaleAdd(T *dst, const U *src, W scale_factor, int32_t n, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < n; i++) {
    dst[i] += src[i] * scale_factor;
  }
}

/// @brief compute multiple density maps of different area types
/// @param pos cell center (x, y) pairs
/// @param node_sizes cell (w, h) pairs
/// @param node_weights weights of the contribution to the density maps; this is
/// for area stretching
/// @param bin_map_dims length of num_bin_maps, record the dimension (x, y) of
/// bin maps
/// @param xl diearea
/// @param yl diearea
/// @param xh diearea
/// @param yh diearea
/// @param bin_w bin width
/// @param bin_h bin height
/// @param range cell index range to be computed for density maps
/// @param num_threads number of threads
/// @param density_maps multiple 2D density maps in row-major to write
template<typename T, typename U, int32_t DensityFunc, typename AtomicOp>
void computeDensityMap(T const            *pos,
        T const                           *node_sizes,
        T const                           *node_weights,
        int32_t const                     *bin_map_dims,
        T                                  xl,
        T                                  yl,
        T                                  xh,
        T                                  yh,
        std::pair<int32_t, int32_t> const &range,
        int32_t                            num_area_types,
        int32_t                            num_threads,
        AtomicOp                           atomic_op,
        U                                **buf_maps) {
  // density_map should be initialized outside

  auto func   = (DensityFunc == 0) ? triangleDensity<T> : exactDensity<T>;
  auto width  = xh - xl;
  auto height = yh - yl;

  for (int32_t i = range.first; i < range.second; ++i) {
    int32_t offset  = (i << 1);
    auto    node_cx = pos[offset];
    auto    node_cy = pos[offset + 1];

    for (int32_t area_type = 0; area_type < num_area_types; ++area_type) {
      offset      = i * num_area_types * 2 + area_type * 2;
      auto node_w = node_sizes[offset];
      auto node_h = node_sizes[offset + 1];
      // skip zeros
      if (node_w * node_h == 0) {
        continue;
      }
      auto node_xl      = node_cx - node_w / 2;
      auto node_yl      = node_cy - node_h / 2;
      auto node_xh      = node_cx + node_w / 2;
      auto node_yh      = node_cy + node_h / 2;
      auto node_weight  = node_weights[(offset >> 1)];

      // get map information for a specific area type
      offset            = (area_type << 1);
      auto    dim_x     = bin_map_dims[offset];
      auto    dim_y     = bin_map_dims[offset + 1];

      auto    buf_map   = buf_maps[area_type];
      auto    bin_w     = width / dim_x;
      auto    bin_h     = height / dim_y;
      auto    inv_bin_w = 1.0 / bin_w;
      auto    inv_bin_h = 1.0 / bin_h;

      int32_t bin_ixl   = int32_t((node_xl - xl) * inv_bin_w);
      // exclusive
      int32_t bin_ixh   = int32_t(((node_xh - xl) * inv_bin_w)) + 1;
      bin_ixl           = OPENPARF_STD_NAMESPACE::max(bin_ixl, 0);
      bin_ixh           = OPENPARF_STD_NAMESPACE::min(bin_ixh, dim_x);

      int32_t bin_iyl   = int32_t((node_yl - yl) * inv_bin_h);
      // exclusive
      int32_t bin_iyh   = int32_t(((node_yh - yl) * inv_bin_h)) + 1;
      bin_iyl           = OPENPARF_STD_NAMESPACE::max(bin_iyl, 0);
      bin_iyh           = OPENPARF_STD_NAMESPACE::min(bin_iyh, dim_y);

      // update density map
      for (int32_t k = bin_ixl; k < bin_ixh; ++k) {
        T bin_xl      = xl + k * bin_w;
        T px          = func(node_xl, node_xh, bin_xl, bin_xl + bin_w);
        T px_by_ratio = px * node_weight;

        for (int32_t h = bin_iyl; h < bin_iyh; ++h) {
          T bin_yl = yl + h * bin_h;
          T py     = func(node_yl, node_yh, bin_yl, bin_yl + bin_h);
          T area   = px_by_ratio * py;
          atomic_op(&buf_map[k * dim_y + h], area);
        }
      }
    }
  }
}

template<typename T, int32_t DensityFunc>
void OPENPARF_NOINLINE computeDensityMapLauncher(T const *pos,
        T const                                          *node_sizes,
        T const                                          *node_weights,
        int32_t const                                    *bin_map_dims,
        T                                                 xl,
        T                                                 yl,
        T                                                 xh,
        T                                                 yh,
        std::pair<int32_t, int32_t> const                &range,
        int32_t                                           num_area_types,
        int32_t                                           num_threads,
        int32_t                                           deterministic_flag,
        std::vector<T *>                                 &density_maps) {
  if (deterministic_flag) {
    double                 width         = xh - xl;
    double                 height        = yh - yl;
    double                 diearea       = width * height;
    int32_t                integer_bits  = OPENPARF_STD_NAMESPACE::max((int32_t) ceil(log2(diearea)) + 1, 32);
    int32_t                fraction_bits = OPENPARF_STD_NAMESPACE::max(64 - integer_bits, 0);
    int64_t                scale_factor  = static_cast<int64_t>(1) << fraction_bits;
    AtomicAdd<int64_t>     atomic_add_op(scale_factor);
    std::vector<int64_t *> buf(num_area_types, nullptr);
    for (int32_t i = 0; i < num_area_types; ++i) {
      int32_t size = bin_map_dims[i << 1] * bin_map_dims[(i << 1) + 1];
      buf[i]       = new int64_t[size];
      memset(buf[i], 0, sizeof(int64_t) * size);
    }
    DEFER({
      for (int32_t i = 0; i < num_area_types; ++i) {
        delete[] buf[i];
      }
    });
    computeDensityMap<T, int64_t, DensityFunc, decltype(atomic_add_op)>(pos, node_sizes, node_weights, bin_map_dims, xl,
            yl, xh, yh, range, num_area_types, num_threads, atomic_add_op, buf.data());
    for (int32_t i = 0; i < num_area_types; ++i) {
      int32_t size = bin_map_dims[i << 1] * bin_map_dims[(i << 1) + 1];
      scaleAdd(density_maps[i], buf[i], 1. / scale_factor, size, num_threads);
    }
  } else {
    AtomicAdd<T> atomic_add_op;
    computeDensityMap<T, T, DensityFunc, decltype(atomic_add_op)>(pos, node_sizes, node_weights, bin_map_dims, xl, yl,
            xh, yh, range, num_area_types, num_threads, atomic_add_op, density_maps.data());
  }
}

template<typename T>
void stretchArea(T const                  *node_sizes,
        int32_t const                     *bin_map_dims,
        T                                  stretch_ratio,
        T                                  xl,
        T                                  yl,
        T                                  xh,
        T                                  yh,
        std::pair<int32_t, int32_t> const &range,
        int32_t                            num_area_types,
        int32_t                            num_threads,
        T                                 *node_sizes_stretched,
        T                                 *node_weights) {
  auto width  = xh - xl;
  auto height = yh - yl;

#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = range.first; i < range.second; ++i) {
    for (int32_t area_type = 0; area_type < num_area_types; ++area_type) {
      int32_t offset = i * num_area_types * 2 + area_type * 2;
      auto    node_w = node_sizes[offset];
      auto    node_h = node_sizes[offset + 1];
      // skip zeros
      if (node_w * node_h == 0) {
        continue;
      }
      auto &node_w_stretched = node_sizes_stretched[offset];
      auto &node_h_stretched = node_sizes_stretched[offset + 1];
      auto &node_weight      = node_weights[(offset >> 1)];

      // get map information for a specific area type
      offset                 = (area_type << 1);
      auto dim_x             = bin_map_dims[offset];
      auto dim_y             = bin_map_dims[offset + 1];
      auto bin_w             = width / dim_x;
      auto bin_h             = height / dim_y;

      node_w_stretched       = OPENPARF_STD_NAMESPACE::max(bin_w * stretch_ratio, node_w);
      node_h_stretched       = OPENPARF_STD_NAMESPACE::max(bin_h * stretch_ratio, node_h);
      node_weight            = (node_w * node_h) / (node_w_stretched * node_h_stretched);
    }
  }
}

template<typename T>
void OPENPARF_NOINLINE stretchAreaLauncher(T const *node_sizes,
        int32_t const                              *bin_map_dims,
        T                                           stretch_ratio,
        T                                           xl,
        T                                           yl,
        T                                           xh,
        T                                           yh,
        std::pair<int32_t, int32_t> const          &range,
        int32_t                                     num_area_types,
        int32_t                                     num_threads,
        T                                          *node_sizes_stretched,
        T                                          *node_weights) {
  stretchArea(node_sizes, bin_map_dims, stretch_ratio, xl, yl, xh, yh, range, num_area_types, num_threads,
          node_sizes_stretched, node_weights);
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void computeDensityMapLauncher<T, 0>(T const *pos, T const *node_sizes, T const *node_weights,              \
          int32_t const *bin_map_dims, T xl, T yl, T xh, T yh, std::pair<int32_t, int32_t> const &range,               \
          int32_t num_area_types, int32_t num_threads, int32_t deterministic_flag, std::vector<T *> &density_maps);    \
  template void computeDensityMapLauncher<T, 1>(T const *pos, T const *node_sizes, T const *node_weights,              \
          int32_t const *bin_map_dims, T xl, T yl, T xh, T yh, std::pair<int32_t, int32_t> const &range,               \
          int32_t num_area_types, int32_t num_threads, int32_t deterministic_flag, std::vector<T *> &density_maps);    \
  template void stretchAreaLauncher<T>(T const *node_sizes, int32_t const *bin_map_dims, T stretch_ratio, T xl, T yl,  \
          T xh, T yh, std::pair<int32_t, int32_t> const &range, int32_t num_area_types, int32_t num_threads,           \
          T *node_sizes_stretched, T *node_weights);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
