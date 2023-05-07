/**
 * File              : direct_lg.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ops/masked_direct_lg/src/direct_lg.h"

// C++ standard library headers
#include <tuple>
#include <vector>

// project headers
#include "database/clock_availability.h"
#include "database/placedb.h"
#include "util/torch.h"

// local headers
#include "ops/masked_direct_lg/src/direct_lg_db.h"
#include "ops/masked_direct_lg/src/direct_lg_kernel.h"
#include "ops/masked_direct_lg/src/dl_problem.h"

OPENPARF_BEGIN_NAMESPACE

/**
 * @brief Directly legalizes the LUT & FF instances without clock constraint for
 * ISPD benchmark, and LUT5/6, LRAM, SHIFT & DFF for Xarch benchmark.
 * @param placedb Placement database
 * @param pyparam Python object recording the parameters
 * @param init_pos  Inital instance position.
 * @return at::Tensor Post-legalization instance position.
 */
at::Tensor directLegalizeForward(database::PlaceDB const &placedb,
                                 py::object               pyparam,
                                 at::Tensor               masked_inst_ids,
                                 at::Tensor               init_pos) {
  using masked_direct_lg::directLegalizeLauncher;
  using masked_direct_lg::DirectLegalizeParam;

  CHECK_FLAT_CPU(masked_inst_ids);
  CHECK_EVEN(masked_inst_ids);
  CHECK_CONTIGUOUS(masked_inst_ids);
  CHECK_FLAT_CPU(init_pos);
  CHECK_DIVISIBLE(init_pos, 3);
  CHECK_CONTIGUOUS(init_pos);

  auto                pos                            = at::zeros({placedb.numInsts(), 3}, init_pos.options());
  DirectLegalizeParam param                          = DirectLegalizeParam::ParseFromPyObject(pyparam);
  param.honorHalfColumnConstraint                    = false;
  param.honorClockRegionConstraint                   = false;
  auto                              hc_avail_map     = at::zeros({1}, torch::dtype(torch::kUInt8));
  int32_t                           num_masked_insts = masked_inst_ids.numel();
  std::vector<std::vector<int32_t>> dummy;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "directLegalizeLauncher", [&] {
    ClockAvailCheckerType<scalar_t> legality_check_functor(
            [](int32_t instance_id, const scalar_t &site_x, const scalar_t &site_y) -> bool { return true; });

    LayoutXy2GridIndexFunctorType<scalar_t> xy_to_half_column_idx_functor = [](scalar_t x, scalar_t y) { return -1; };

    directLegalizeLauncher<scalar_t>(placedb,
                                     param,
                                     OPENPARF_TENSOR_DATA_PTR(init_pos, scalar_t),
                                     legality_check_functor,
                                     xy_to_half_column_idx_functor,
                                     dummy,
                                     at::get_num_threads(),
                                     num_masked_insts,
                                     OPENPARF_TENSOR_DATA_PTR(masked_inst_ids, int32_t),
                                     OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                     OPENPARF_TENSOR_DATA_PTR(hc_avail_map, uint8_t));
  });
  return pos;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::directLegalizeForward, "Direct legalization forward");
}
