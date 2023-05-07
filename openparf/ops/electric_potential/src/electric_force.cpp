/**
 * @file   electric_force.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Compute electric force according to e-place
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeElectricForceLauncher(
    T const *grad_pos, T const *pos, T const *node_sizes, T const *node_weights,
    std::vector<T *> &field_map_xs,
    std::vector<T *> &field_map_ys, int32_t *bin_map_dims, double xl, double yl,
    double xh, double yh, std::pair<int32_t, int32_t> range,
    int32_t num_area_types,
    int32_t num_threads, T *grad_out);

#define CALL_LAUNCHER(range)                                                 \
  computeElectricForceLauncher<scalar_t>(                                    \
      OPENPARF_TENSOR_DATA_PTR(grad_pos, scalar_t),                          \
      OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),                               \
      OPENPARF_TENSOR_DATA_PTR(node_sizes, scalar_t),                        \
      OPENPARF_TENSOR_DATA_PTR(node_weights, scalar_t),                      \
      field_map_x_ptrs,  \
      field_map_y_ptrs, OPENPARF_TENSOR_DATA_PTR(bin_map_dims, int32_t), xl, \
      yl, xh, yh, range, num_area_types, at::get_num_threads(),                              \
      OPENPARF_TENSOR_DATA_PTR(grad_out, scalar_t))

/// @brief compute electric force for movable and filler cells
/// @param grad_pos input gradient from backward propagation
at::Tensor electricForce(at::Tensor grad_pos, at::Tensor pos,
                         at::Tensor node_sizes, at::Tensor node_weights,
                         std::vector<at::Tensor> const &field_map_xs,
                         std::vector<at::Tensor> const &field_map_ys,
                         at::Tensor bin_map_dims, double xl, double yl,
                         double xh, double yh,
                         std::pair<int32_t, int32_t> movable_range,
                         std::pair<int32_t, int32_t> filler_range,
                         int32_t num_area_types) {
  CHECK_FLAT_CPU(grad_pos);
  CHECK_CONTIGUOUS(grad_pos);
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(node_sizes);
  CHECK_EVEN(node_sizes);
  CHECK_CONTIGUOUS(node_sizes);
  CHECK_FLAT_CPU(node_weights);
  CHECK_CONTIGUOUS(node_weights);
  for (auto const &field_map_x : field_map_xs) {
    CHECK_FLAT_CPU(field_map_x);
    CHECK_CONTIGUOUS(field_map_x);
  }
  for (auto const &field_map_y : field_map_ys) {
    CHECK_FLAT_CPU(field_map_y);
    CHECK_CONTIGUOUS(field_map_y);
  }

  at::Tensor grad_out = at::zeros_like(pos);

  OPENPARF_DISPATCH_FLOATING_TYPES(
      pos, "computeElectricForceLauncher", [&] {
        std::vector<scalar_t *> field_map_x_ptrs(field_map_xs.size());
        std::vector<scalar_t *> field_map_y_ptrs(field_map_xs.size());
        for (uint32_t i = 0; i < field_map_xs.size(); ++i) {
          field_map_x_ptrs[i] =
              OPENPARF_TENSOR_DATA_PTR(field_map_xs[i], scalar_t);
          field_map_y_ptrs[i] =
              OPENPARF_TENSOR_DATA_PTR(field_map_ys[i], scalar_t);
        }
        if (movable_range.first < movable_range.second) {
          CALL_LAUNCHER(movable_range);
        }
        if (filler_range.first < filler_range.second) {
          CALL_LAUNCHER(filler_range);
        }
      });

  return grad_out;
}

#undef CALL_LAUNCHER

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &OPENPARF_NAMESPACE::electricForce, "Backward propagation");
}
