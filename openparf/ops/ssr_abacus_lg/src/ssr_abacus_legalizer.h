/**
 * File              : ssr_abacus_legalizer.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_SSR_ABACUS_LG_SRC_CHAIN_LEGALIZER_H_
#define OPENPARF_OPS_SSR_ABACUS_LG_SRC_CHAIN_LEGALIZER_H_
#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace ssr_abacus_legalizer {
void SsrLegalizerForward(database::PlaceDB const &placedb,
                         at::Tensor               pos,
                         at::Tensor               concerned_inst_ids,
                         int32_t                  area_type_id,
                         at::Tensor               chain_ssr_ids_bs,
                         at::Tensor               chain_ssr_ids_b_starts);
}
OPENPARF_END_NAMESPACE


#endif   // OPENPARF_OPS_SSR_ABACUS_LG_SRC_CHAIN_LEGALIZER_H_
