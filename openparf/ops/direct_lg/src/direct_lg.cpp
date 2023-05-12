/**
 * File              : direct_lg.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ops/direct_lg/src/direct_lg.h"

// C++ standard library headers
#include <tuple>
#include <vector>

// project headers
#include "database/clock_availability.h"
#include "database/placedb.h"
#include "util/torch.h"

// local headers
#include "ops/direct_lg/src/direct_lg_db.h"
#include "ops/direct_lg/src/direct_lg_kernel.h"
#include "ops/direct_lg/src/dl_problem.h"

OPENPARF_BEGIN_NAMESPACE

/**
 * @brief Directly legalizes the LUT & FF instances without clock constraint for
 * ISPD benchmark, and LUT5/6, LRAM, SHIFT & DFF for Xarch benchmark.
 * @param placedb Placement database
 * @param pyparam Python object recording the parameters
 * @param init_pos  Inital instance position.
 * @return at::Tensor Post-legalization instance position.
 */
at::Tensor directLegalizeForward(database::PlaceDB const &placedb, py::object pyparam, at::Tensor init_pos) {
  using direct_lg::directLegalizeLauncher;
  using direct_lg::DirectLegalizeParam;

  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto                pos                        = at::zeros({placedb.numInsts(), 3}, init_pos.options());
  DirectLegalizeParam param                      = DirectLegalizeParam::ParseFromPyObject(pyparam);
  param.honorHalfColumnConstraint                = false;
  param.honorClockRegionConstraint               = false;
  auto                              hc_avail_map = at::zeros({1}, torch::dtype(torch::kUInt8));
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
                                     OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                     OPENPARF_TENSOR_DATA_PTR(hc_avail_map, uint8_t));
  });
  return pos;
}

/**
 * @brief Directly legalizes the LUT & FF instances with clock constraint for
 * ISPD benchmark.
 * @param placedb Placement database
 * @param pyparam Python object recording the parameters
 * @param init_pos  Inital instance position.
 * @return at::Tensor Post-legalization instance position.
 */
std::tuple<at::Tensor, at::Tensor> ClockAwareDirectLegalizeForward(
        database::PlaceDB const &                placedb,
        py::object                               pyparam,
        at::Tensor                               init_pos,
        const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
        at::Tensor                               clock_available_clock_region) {
  using direct_lg::directLegalizeLauncher;
  using direct_lg::DirectLegalizeParam;

  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  CHECK_FLAT_CPU(clock_available_clock_region);
  CHECK_CONTIGUOUS(clock_available_clock_region);

  auto                pos   = at::zeros({placedb.numInsts(), 3}, init_pos.options());
  DirectLegalizeParam param = DirectLegalizeParam::ParseFromPyObject(pyparam);
  auto hc_avail_map = at::zeros({placedb.numClockNets(), placedb.numHalfColumnRegions()}, torch::dtype(torch::kUInt8));
  param.honorHalfColumnConstraint  = true;
  param.honorClockRegionConstraint = true;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "directLegalizeLauncher", [&] {
    LayoutXy2GridIndexFunctorType<scalar_t> xy_to_cr_idx = [&](scalar_t x, scalar_t y) {
      return placedb.XyToCrIndex(x, y);
    };
    LayoutXy2GridIndexFunctorType<scalar_t> xy_to_half_column_idx_functor = [&](scalar_t x, scalar_t y) {
      return placedb.XyToHcIndex(x, y);
    };

    ClockAvailCheckerType<scalar_t> legality_check_functor =
            GenClockAvailChecker<scalar_t>(inst_to_clock_indexes, clock_available_clock_region, xy_to_cr_idx);
    directLegalizeLauncher<scalar_t>(placedb,
                                     param,
                                     OPENPARF_TENSOR_DATA_PTR(init_pos, scalar_t),
                                     legality_check_functor,
                                     xy_to_half_column_idx_functor,
                                     inst_to_clock_indexes,
                                     at::get_num_threads(),
                                     OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                     OPENPARF_TENSOR_DATA_PTR(hc_avail_map, uint8_t));
  });
  return {pos, hc_avail_map};
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::directLegalizeForward, "Direct legalization forward")
          .def("clock_aware_forward",
               &OPENPARF_NAMESPACE::ClockAwareDirectLegalizeForward,
               "Clock-aware direct legalization forward");
}
