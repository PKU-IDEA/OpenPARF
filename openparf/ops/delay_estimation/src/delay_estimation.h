#ifndef OPENPARF_OPS_DELAY_ESTIMATION_SRC_DELAY_ESTIMATION_H_
#define OPENPARF_OPS_DELAY_ESTIMATION_SRC_DELAY_ESTIMATION_H_

// C++ library headers
#include <tuple>

// Project headers
#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"


OPENPARF_BEGIN_NAMESPACE

namespace delay_estimation {
std::tuple<at::Tensor, at::Tensor> DelayEstimationForward(database::PlaceDB const& placedb,
                                                          at::Tensor               net_mask_ignore_large,
                                                          at::Tensor               pos);

}

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_DELAY_ESTIMATION_SRC_DELAY_ESTIMATION_H_
