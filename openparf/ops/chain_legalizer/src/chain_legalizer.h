/**
 * File              : chain_legalizer.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.10.2021
 * Last Modified Date: 09.10.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_CHAIN_LEGALIZER_SRC_CHAIN_LEGALIZER_H_
#define OPENPARF_OPS_CHAIN_LEGALIZER_SRC_CHAIN_LEGALIZER_H_

#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace chain_legalizer {
void ChainLegalizerForward(database::PlaceDB const &placedb,
                           at::Tensor               pos_xyz,
                           at::Tensor               concerned_inst_ids,
                           int32_t                  area_type_id,
                           int32_t                  search_manh_dist_increment,
                           int32_t                  max_iter,
                           at::Tensor               chain_cla_ids_bs,
                           at::Tensor               chain_cla_ids_b_starts,
                           at::Tensor               chain_lut_ids_bs,
                           at::Tensor               chain_lut_ids_b_starts);
}
OPENPARF_END_NAMESPACE


#endif   // OPENPARF_OPS_CHAIN_LEGALIZER_SRC_CHAIN_LEGALIZER_H_
