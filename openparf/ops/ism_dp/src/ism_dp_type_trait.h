/**
 * File              : ism_dp_type_trait.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.05.2021
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DP_TYPE_TRAIT_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DP_TYPE_TRAIT_H_

// C++ standard library headers
#include <cstdint>
#include <limits>

// project headers
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

using Byte                        = std::uint8_t;
using IndexType                   = std::int32_t;
using RealType                    = float;

constexpr IndexType kIndexTypeMax = std::numeric_limits<IndexType>::max();
constexpr RealType  kRealTypeMax  = std::numeric_limits<RealType>::max();
constexpr RealType  kRealTypeMin  = std::numeric_limits<RealType>::lowest();

}   // namespace ism_dp

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DP_TYPE_TRAIT_H_
