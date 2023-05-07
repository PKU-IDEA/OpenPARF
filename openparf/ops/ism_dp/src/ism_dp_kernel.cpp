/**
 * File              : ism_dp_kernel.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.01.2020
 * Last Modified Date: 07.01.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#include "ops/ism_dp/src/ism_dp_kernel.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void ismDetailedPlaceLauncher<T>(database::PlaceDB const&                          db,                      \
                                            ISMDetailedPlaceParam                             param,                   \
                                            std::function<bool(uint32_t, uint32_t, uint32_t)> isCKAllowedInSite,       \
                                            bool                                              honorClockConstraint,    \
                                            int32_t                                           num_threads,             \
                                            T*                                                pos);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace ism_dp

OPENPARF_END_NAMESPACE
