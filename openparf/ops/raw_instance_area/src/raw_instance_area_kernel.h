/**
 * File              : raw_instance_area_kernel.h
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.25.2020
 * Last Modified Date: 07.25.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */
#ifndef OPENPARF_RAW_INSTANCE_AREA_KERNEL_H
#define OPENPARF_RAW_INSTANCE_AREA_KERNEL_H

#include "util/util.h"
// local dependency
#include "scaling_function.h"

OPENPARF_BEGIN_NAMESPACE

template<typename DataType, typename IndexType>
int computeAdjustedInstanceAreaLauncher(
        DataType *cell_pos,
        DataType *cell_half_sizes,
        std::pair<IndexType, IndexType> movable_range,
        DataType *utilization_map,
        IndexType xl,
        IndexType yl,
        IndexType xh,
        IndexType yh,
        IndexType num_bins_x,
        IndexType num_bins_y,
        DataType bin_size_x,
        DataType bin_size_y,
        IndexType num_threads,
        DataType *adjusted_instance_area) {
    DataType inv_bin_size_x = 1.0 / bin_size_x;
    DataType inv_bin_size_y = 1.0 / bin_size_y;
    DataType *movable_cell_pos = cell_pos + (2 * movable_range.first);
    DataType *movable_cell_half_sizes = cell_half_sizes + (2 * movable_range.first);

    IndexType num_movable_cells = movable_range.second - movable_range.first;

    IndexType chunk_size = OPENPARF_STD_NAMESPACE::max(int(num_movable_cells / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (IndexType i = 0; i < num_movable_cells; i++) {
        IndexType idx_x = i << 1;
        IndexType idx_y = idx_x | 1;
        DataType center_x = movable_cell_pos[idx_x];
        DataType center_y = movable_cell_pos[idx_y];
        DataType half_width = movable_cell_half_sizes[idx_x];
        DataType half_height = movable_cell_half_sizes[idx_y];

        // ignore the cell whose area is equal to zero.
        if (half_width < std::numeric_limits<DataType>::epsilon()
            || half_height < std::numeric_limits<DataType>::epsilon()){
            adjusted_instance_area[i] = 0;
            continue;
        }

        DataType x_min = center_x - half_width;
        DataType x_max = center_x + half_width;
        DataType y_min = center_y - half_height;
        DataType y_max = center_y + half_height;

        // compute the bin box that this instance covers
        DataType bin_index_xl = DataType((x_min - xl) * inv_bin_size_x);
        DataType bin_index_xh = DataType((x_max - xl) * inv_bin_size_x) + 1;
        bin_index_xl = OPENPARF_STD_NAMESPACE::max(bin_index_xl, (DataType) 0);
        bin_index_xh = OPENPARF_STD_NAMESPACE::min(bin_index_xh, (DataType) num_bins_x);
        DataType bin_index_yl = DataType((y_min - yl) * inv_bin_size_y);
        DataType bin_index_yh = DataType((y_max - yl) * inv_bin_size_y) + 1;
        bin_index_yl = OPENPARF_STD_NAMESPACE::max(bin_index_yl, (DataType) 0);
        bin_index_yh = OPENPARF_STD_NAMESPACE::min(bin_index_yh, (DataType) num_bins_y);

        adjusted_instance_area[i] = averageScaling<DataType, IndexType>(
                utilization_map,
                xl, yl,
                bin_size_x, bin_size_y,
                num_bins_x, num_bins_y,
                bin_index_xl,
                bin_index_yl,
                bin_index_xh,
                bin_index_yh,
                x_min, y_min, x_max, y_max);
    }
    return 0;
}

OPENPARF_END_NAMESPACE

#endif//OPENPARF_RAW_INSTANCE_AREA_KERNEL_H
