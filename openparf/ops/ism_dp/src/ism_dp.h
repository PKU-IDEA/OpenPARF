/**
 * File              : ism_dp.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.14.2021
 * Last Modified Date: 09.14.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DP_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DP_H_

// project headers
#include "database/placedb.h"
#include "util/torch.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

/// @param init_pos cell locations, array of (x, y) pairs
at::Tensor ismDetailedPlaceForward(database::PlaceDB const& placedb,
                                   py::object               pyparam,
                                   at::Tensor               clock_available_clock_region,
                                   at::Tensor               hc_available_clock_region,
                                   at::Tensor               fixed_mask,
                                   at::Tensor               init_pos);
}   // namespace ism_dp

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DP_H_
