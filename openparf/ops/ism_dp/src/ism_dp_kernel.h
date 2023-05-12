/**
 * File              : ism_dp_kernel.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.13.2021
 * Last Modified Date: 09.13.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DP_KERNEL_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DP_KERNEL_H_

#include "ops/ism_dp/src/ism_detailed_placer.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

template<typename T>
void ismDetailedPlaceLauncher(database::PlaceDB const&                          db,
                              ISMDetailedPlaceParam                             param,
                              std::function<bool(uint32_t, uint32_t, uint32_t)> isCKAllowedInSite,
                              bool                                              honorClockConstraint,
                              int32_t                                           num_threads,
                              T*                                                pos) {
  ISMDetailedPlacer<T> placer(db, param, isCKAllowedInSite, honorClockConstraint, num_threads);
  placer.run(pos);
}
}   // namespace ism_dp

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DP_KERNEL_H_
