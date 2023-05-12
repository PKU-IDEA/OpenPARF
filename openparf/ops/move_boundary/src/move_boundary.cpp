/**
 * @file   move_boundary.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Move out-of-bound cells back to inside placement region
 */
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeMoveBoundaryLauncher(T* pos, T const* node_sizes, T xl, T yl, T xh,
                                 T yh, std::pair<int32_t, int32_t> const& range,
                                 int32_t num_threads);

#define CALL_LAUNCHER(range)                                                 \
  computeMoveBoundaryLauncher<scalar_t>(                                     \
      OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),                               \
      OPENPARF_TENSOR_DATA_PTR(node_sizes, scalar_t), xl, yl, xh, yh, range, \
      at::get_num_threads())

at::Tensor moveBoundaryForward(at::Tensor pos, at::Tensor node_sizes, double xl,
                               double yl, double xh, double yh,
                               std::pair<int32_t, int32_t> movable_range,
                               std::pair<int32_t, int32_t> filler_range) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(node_sizes);
  CHECK_EVEN(node_sizes);
  CHECK_CONTIGUOUS(node_sizes);

  OPENPARF_DISPATCH_FLOATING_TYPES(
      pos, "computeMoveBoundaryLauncher", [&] {
        if (movable_range.first < movable_range.second) {
          CALL_LAUNCHER(movable_range);
        }
        if (filler_range.first < filler_range.second) {
          CALL_LAUNCHER(filler_range);
        }
      });

  return pos;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::moveBoundaryForward,
        "MoveBoundary forward");
}
