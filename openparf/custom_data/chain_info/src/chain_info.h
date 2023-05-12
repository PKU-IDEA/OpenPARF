#ifndef OPENPARF_CUSTOM_DATA_CHAIN_INFO_SRC_CHAIN_INFO_H_
#define OPENPARF_CUSTOM_DATA_CHAIN_INFO_SRC_CHAIN_INFO_H_
#include <tuple>
#include <vector>

#include "database/placedb.h"
#include "pybind/container.h"
#include "pybind/util.h"
#include "util/arg.h"
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace chain_info {

/**@
 * @brief The info collection for a single carry chain specially for xarch. Note that under xarch's rules, a chain not
 * noly consists of CLAs cells, but also constains associated LUTs. See xarch's architecture for further explanation.
 *
 */
class ChainInfo {
  CLASS_ARG(VectorInt, cla_ids);       ///< The CLA cell ids from low-bit to high-bit.
  CLASS_ARG(VectorInt, lut_ids);       ///< The LUT cell ids from low-bit to high-bit.
  CLASS_ARG(VectorInt, cla_new_ids);   ///< The *new* CLA cell ids from low-bit to high-bit.
  CLASS_ARG(VectorInt, lut_new_ids);   ///< The *new* LUT cell ids from low-bit to high-bit.
};

using ChainInfoVec = std::vector<ChainInfo>;

ChainInfoVec                                         MakeChainInfoVecFromPlaceDB(const database::PlaceDB &placedb);

std::tuple<FlatNestedVectorInt, FlatNestedVectorInt> MakeNestedNewIdx(const ChainInfoVec &chain_info_vec);


namespace detail {

ChainInfo ExtractChainFromOneInst(const database::PlaceDB &placedb,
                                  int32_t                  module_id,
                                  const database::Inst &   source_inst,
                                  std::vector<bool> &      visited_mark);
}   // namespace detail

}   // namespace chain_info


OPENPARF_END_NAMESPACE

#endif   // OPENPARF_CUSTOM_DATA_CHAIN_INFO_SRC_CHAIN_INFO_H_
