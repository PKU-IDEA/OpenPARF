/**
 * File              : raw_instance_area.cpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.24.2020
 * Last Modified Date: 07.25.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"
// local dependency
#include "raw_instance_area_kernel.h"

OPENPARF_BEGIN_NAMESPACE

at::Tensor raw_instance_area_forward(
        at::Tensor cell_pos,
        at::Tensor cell_half_sizes,
        std::pair<int32_t, int32_t> movable_range,
        at::Tensor utilization_map,
        double xl,
        double yl,
        double xh,
        double yh,
        int32_t num_bins_x,
        int32_t num_bins_y,
        double bin_size_x,
        double bin_size_y) {
    CHECK_FLAT_CPU(cell_pos);
    CHECK_EVEN(cell_pos);
    CHECK_CONTIGUOUS(cell_pos);

    CHECK_FLAT_CPU(cell_half_sizes);
    CHECK_EVEN(cell_half_sizes);
    CHECK_CONTIGUOUS(cell_half_sizes);

    CHECK_FLAT_CPU(utilization_map);
    CHECK_CONTIGUOUS(utilization_map);

    int32_t num_cells = cell_pos.numel() / 2;
    openparfAssert(num_cells == cell_half_sizes.numel() / 2);
    int32_t num_movable_cells = movable_range.second - movable_range.first;
    at::Tensor adjusted_instance_area = at::zeros({num_movable_cells}, cell_pos.options());

    OPENPARF_DISPATCH_FLOATING_TYPES(cell_pos, "computeAdjustedInstanceAreaLauncher", [&] {
        computeAdjustedInstanceAreaLauncher<scalar_t, int32_t>(
                OPENPARF_TENSOR_DATA_PTR(cell_pos, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(cell_half_sizes, scalar_t),
                movable_range,
                OPENPARF_TENSOR_DATA_PTR(utilization_map, scalar_t),
                xl, yl, xh, yh, num_bins_x, num_bins_y, bin_size_x, bin_size_y, at::get_num_threads(),
                OPENPARF_TENSOR_DATA_PTR(adjusted_instance_area, scalar_t));
    });
    return adjusted_instance_area;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::raw_instance_area_forward, "Compute adjusted area for routability optimization");
}
