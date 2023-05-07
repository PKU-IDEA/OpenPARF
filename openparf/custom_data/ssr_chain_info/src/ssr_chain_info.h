/**
 * File              : ssr_chain_info.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_CUSTOM_DATA_SSR_CHAIN_INFO_SRC_SSR_CHAIN_INFO_H_
#define OPENPARF_CUSTOM_DATA_SSR_CHAIN_INFO_SRC_SSR_CHAIN_INFO_H_
#include <tuple>
#include <vector>

#include "database/placedb.h"
#include "pybind/container.h"
#include "pybind/util.h"
#include "util/arg.h"
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace ssr_chain_info {

/**@
 * @brief The info collection for a single GCU carry chain specially for xarch.
 *
 */
class ChainInfo {
  CLASS_ARG(VectorInt, inst_ids);       ///< The instance ids from low-bit to high-bit.
  CLASS_ARG(VectorInt, inst_new_ids);   ///< The *new* instance ids from low-bit to high-bit.
};

using ChainInfoVec = std::vector<ChainInfo>;

ChainInfoVec        MakeChainInfoVecFromPlaceDB(const database::PlaceDB &placedb);

FlatNestedVectorInt MakeNestedNewIdx(const ChainInfoVec &chain_info_vec);

namespace detail {

ChainInfo ExtractChainFromOneInst(const database::PlaceDB &placedb,
                                  int32_t                  module_id,
                                  const database::Inst    &source_inst,
                                  std::vector<bool>       &visited_mark);
}   // namespace detail

}   // namespace ssr_chain_info


OPENPARF_END_NAMESPACE

#endif   // OPENPARF_CUSTOM_DATA_SSR_CHAIN_INFO_SRC_SSR_CHAIN_INFO_H_
