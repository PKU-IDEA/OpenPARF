/**
 * File              : scaling_function.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.24.2020
 * Last Modified Date: 07.24.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */
#ifndef OPENPARF_SCALING_FUNCTION_H
#define OPENPARF_SCALING_FUNCTION_H

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template<typename DataType, typename IndexType>
inline DataType averageScaling(
        const DataType *utilization_map,
        DataType xl, DataType yl,
        DataType bin_size_x, DataType bin_size_y,
        IndexType num_bins_x, IndexType num_bins_y,
        IndexType bin_index_xl,
        IndexType bin_index_yl,
        IndexType bin_index_xh,
        IndexType bin_index_yh,
        DataType x_min, DataType y_min, DataType x_max, DataType y_max) {
    DataType area = 0;
    for (IndexType x = bin_index_xl; x < bin_index_xh; ++x) {
        for (IndexType y = bin_index_yl; y < bin_index_yh; ++y) {
            DataType bin_xl = xl + x * bin_size_x;
            DataType bin_yl = yl + y * bin_size_y;
            DataType bin_xh = bin_xl + bin_size_x;
            DataType bin_yh = bin_yl + bin_size_y;
            DataType overlap = OPENPARF_STD_NAMESPACE::max(
                                       OPENPARF_STD_NAMESPACE::min(x_max, bin_xh) - OPENPARF_STD_NAMESPACE::max(x_min, bin_xl),
                                       (DataType) 0) *
                               OPENPARF_STD_NAMESPACE::max(
                                       OPENPARF_STD_NAMESPACE::min(y_max, bin_yh) - OPENPARF_STD_NAMESPACE::max(y_min, bin_yl),
                                       (DataType) 0);
            area += overlap * utilization_map[x * num_bins_y + y];
        }
    }
    return area;
}

template<typename DataType, typename IndexType>
inline DataType maxScaling(
        const DataType *utilization_map,
        DataType xl, DataType yl,
        DataType bin_size_x, DataType bin_size_y,
        IndexType num_bins_x, IndexType num_bins_y,
        IndexType bin_index_xl,
        IndexType bin_index_yl,
        IndexType bin_index_xh,
        IndexType bin_index_yh,
        DataType x_min, DataType y_min, DataType x_max, DataType y_max) {
    DataType util = 0;
    for (IndexType x = bin_index_xl; x < bin_index_xh; ++x) {
        for (IndexType y = bin_index_yl; y < bin_index_yh; ++y) {
            util = OPENPARF_STD_NAMESPACE::max(util, utilization_map[x * num_bins_y + y]);
        }
    }
    DataType area = (x_max - x_min) * (y_max - y_min);
    return area * util;
}

OPENPARF_END_NAMESPACE

#endif//OPENPARF_SCALING_FUNCTION_H
