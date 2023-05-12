/**
 * File              : io_legalizer.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 08.26.2021
 * Last Modified Date: 08.26.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

void IoLegalizerForward(database::PlaceDB const &placedb,
                        at::Tensor               pos,
                        at::Tensor               pos_xyz,
                        at::Tensor               concerned_inst_ids,
                        int32_t                  area_type_id);

OPENPARF_END_NAMESPACE
