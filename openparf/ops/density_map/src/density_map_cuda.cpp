/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Compute density map on GPU
 */
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template<typename T, int32_t DensityFunc>
void computeDensityMapCudaLauncher(T const *pos,
        T const                            *node_sizes,
        T const                            *node_weights,
        int32_t const                      *bin_map_dims,
        T                                   xl,
        T                                   yl,
        T                                   xh,
        T                                   yh,
        std::pair<int32_t, int32_t> const  &range,
        int32_t                             num_area_types,
        int32_t                             deterministic_flag,
        std::vector<T *>                   &density_maps);

/// @brief compute multiple density maps of different area types
/// @tparam DensityFunc 0 for triangle density, 1 for exact density
/// @param pos cell center (x, y) pairs
/// @param node_sizes cell (w, h) * num_area_types pairs
/// @param node_weights weights of the contribution to the density maps; this is
/// for area stretching
/// @param density_maps multiple 2D density maps in row-major to write
/// @param bin_map_dims length of num_bin_maps, record the dimension (x, y) of
/// bin maps
/// @param xl diearea
/// @param yl diearea
/// @param xh diearea
/// @param yh diearea
/// @param bin_w bin width
/// @param bin_h bin height
/// @param range cell index range to be computed for density maps
template<int32_t DensityFunc>
void densityMap(at::Tensor          pos,
        at::Tensor                  node_sizes,
        at::Tensor                  node_weights,
        at::Tensor                  bin_map_dims,
        double                      xl,
        double                      yl,
        double                      xh,
        double                      yh,
        std::pair<int32_t, int32_t> range,
        int32_t                     num_area_types,
        int32_t                     deterministic_flag,
        std::vector<at::Tensor>    &density_maps) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(node_sizes);
  CHECK_EVEN(node_sizes);
  CHECK_CONTIGUOUS(node_sizes);
  CHECK_FLAT_CUDA(node_weights);
  CHECK_CONTIGUOUS(node_weights);
  CHECK_FLAT_CUDA(bin_map_dims);
  CHECK_CONTIGUOUS(bin_map_dims);
  for (auto const &density_map : density_maps) {
    CHECK_FLAT_CUDA(density_map);
    CHECK_CONTIGUOUS(density_map);
  }

  if (range.first < range.second) {
    OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeDensityMapCudaLauncher", [&] {
      std::vector<scalar_t *> density_map_ptrs(density_maps.size());
      for (uint32_t i = 0; i < density_maps.size(); ++i) {
        density_map_ptrs[i] = OPENPARF_TENSOR_DATA_PTR(density_maps[i], scalar_t);
      }
      computeDensityMapCudaLauncher<scalar_t, DensityFunc>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
              OPENPARF_TENSOR_DATA_PTR(node_sizes, scalar_t), OPENPARF_TENSOR_DATA_PTR(node_weights, scalar_t),
              OPENPARF_TENSOR_DATA_PTR(bin_map_dims, int32_t), xl, yl, xh, yh, range, num_area_types,
              deterministic_flag, density_map_ptrs);
    });
  }
}

/// @brief Compute density maps for different area types.
/// @param movable_range range for movable cells
/// @param filler_range range for fixed cells
void movableDensityMapForward(at::Tensor pos,
        at::Tensor                       node_sizes,
        at::Tensor                       node_weights,
        at::Tensor                       bin_map_dims,
        double                           xl,
        double                           yl,
        double                           xh,
        double                           yh,
        std::pair<int32_t, int32_t>      movable_range,
        std::pair<int32_t, int32_t>      filler_range,
        int32_t                          num_area_types,
        int32_t                          deterministic_flag,
        std::vector<at::Tensor>         &density_maps   ///< both input and output
) {
  densityMap<0>(pos, node_sizes, node_weights, bin_map_dims, xl, yl, xh, yh, movable_range, num_area_types,
          deterministic_flag, density_maps);
  densityMap<0>(pos, node_sizes, node_weights, bin_map_dims, xl, yl, xh, yh, filler_range, num_area_types,
          deterministic_flag, density_maps);
}

/// @brief Compute the density maps for fixed cells.
/// @param fixed_range range for fixed cells
void fixedDensityMapForward(at::Tensor pos,
        at::Tensor                     node_sizes,
        at::Tensor                     node_weights,
        at::Tensor                     bin_map_dims,
        double                         xl,
        double                         yl,
        double                         xh,
        double                         yh,
        std::pair<int32_t, int32_t>    fixed_range,
        int32_t                        num_area_types,
        int32_t                        deterministic_flag,
        std::vector<at::Tensor>       &density_maps   ///< both input and output
) {
  densityMap<1>(pos, node_sizes, node_weights, bin_map_dims, xl, yl, xh, yh, fixed_range, num_area_types,
          deterministic_flag, density_maps);
}

template<typename T>
void stretchAreaCudaLauncher(T const      *node_sizes,
        int32_t const                     *bin_map_dims,
        T                                  stretch_ratio,
        T                                  xl,
        T                                  yl,
        T                                  xh,
        T                                  yh,
        std::pair<int32_t, int32_t> const &range,
        int32_t                            num_area_types,
        T                                 *node_sizes_stretched,
        T                                 *node_weights);

#define CALL_LAUNCHER(range)                                                                                           \
  stretchAreaCudaLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(node_sizes, scalar_t),                                    \
          OPENPARF_TENSOR_DATA_PTR(bin_map_dims, int32_t), stretch_ratio, xl, yl, xh, yh, range, num_area_types,       \
          OPENPARF_TENSOR_DATA_PTR(node_sizes_stretched, scalar_t), OPENPARF_TENSOR_DATA_PTR(node_weights, scalar_t))

/// @brief Compute density maps for different area types.
/// @param range range for cells
/// @param node_sizes_stretched output stretched cell sizes
/// @param node_weights output ratio of original area over stretched area
void stretchAreaForward(at::Tensor  node_sizes,
        at::Tensor                  bin_map_dims,
        double                      stretch_ratio,
        double                      xl,
        double                      yl,
        double                      xh,
        double                      yh,
        std::pair<int32_t, int32_t> movable_range,
        std::pair<int32_t, int32_t> filler_range,
        int32_t                     num_area_types,
        at::Tensor                  node_sizes_stretched,
        at::Tensor                  node_weights) {
  CHECK_FLAT_CUDA(node_sizes);
  CHECK_FLAT_CUDA(bin_map_dims);
  CHECK_FLAT_CUDA(node_sizes_stretched);
  CHECK_FLAT_CUDA(node_weights);
  OPENPARF_DISPATCH_FLOATING_TYPES(node_sizes, "stretchAreaCudaLauncher", [&] {
    if (movable_range.first < movable_range.second) {
      CALL_LAUNCHER(movable_range);
    }
    if (filler_range.first < filler_range.second) {
      CALL_LAUNCHER(filler_range);
    }
  });
}

#undef CALL_LAUNCHER

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("movableForward", &OPENPARF_NAMESPACE::movableDensityMapForward,
          "DensityMap for movable and filler cells (CUDA)");
  m.def("fixedForward", &OPENPARF_NAMESPACE::fixedDensityMapForward, "DensityMap for fixed cells (CUDA)");
  m.def("stretchForward", &OPENPARF_NAMESPACE::stretchAreaForward, "Stretch cell sizes (CUDA)");
}
