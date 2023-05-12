#ifndef OPENPARF_OPS_STATIC_TIMING_ANALYSIS_SRC_STATIC_TIMING_ANALYSIS_H_
#define OPENPARF_OPS_STATIC_TIMING_ANALYSIS_SRC_STATIC_TIMING_ANALYSIS_H_

// c++ libraries headers
#include <tuple>

// project headers
#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace static_timing_analysis {

std::tuple<at::Tensor, at::Tensor, at::Tensor> StaticTimingAnalysisForward(database::PlaceDB const& placedb,
                                                                           at::Tensor               pin_delays,
                                                                           at::Tensor net_mask_ignore_large,
                                                                           double                   timing_period);

}   // namespace static_timing_analysis

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_STATIC_TIMING_ANALYSIS_SRC_STATIC_TIMING_ANALYSIS_H_
