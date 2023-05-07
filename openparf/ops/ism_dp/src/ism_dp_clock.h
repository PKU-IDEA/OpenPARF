/**
 * File              : ism_dp_clock.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.13.2021
 * Last Modified Date: 09.13.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DP_CLOCK_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DP_CLOCK_H_

// C++ standard library headers
#include <functional>
#include <vector>

// project headers
#include "database/clock_availability.h"
#include "database/placedb.h"
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

class ISMDetailedPlaceClockDataBase {
 public:
  using IndexType                    = database::PlaceDB::IndexType;
  using CoordinateType               = database::PlaceDB::CoordinateType;

  using LayoutXy2RegionIdFunctorType = std::function<IndexType(const CoordinateType &, const CoordinateType &)>;
  using Vector2DIndexFunctorType     = std::function<const std::vector<std::vector<IndexType>> &()>;

  LayoutXy2RegionIdFunctorType               layout_to_cr_id_functor;
  LayoutXy2RegionIdFunctorType               layout_to_hc_id_functor;
  ClockIndex2RegionIndexCheckerType<int32_t> clock_to_cr_avail;
  ClockIndex2RegionIndexCheckerType<int32_t> clock_to_hc_avail;
  Vector2DIndexFunctorType                   inst_to_clock_indexes;
};


}   // namespace ism_dp
OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DP_CLOCK_H_

