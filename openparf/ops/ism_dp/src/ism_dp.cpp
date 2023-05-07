/**
 * File              : ism_dp.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.01.2020
 * Last Modified Date: 07.01.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#include "ops/ism_dp/src/ism_dp.h"

// C++ standard library headers
#include <functional>

// project headers
#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

// local headers
#include "ops/ism_dp/src/ism_dp_db.hpp"
#include "ops/ism_dp/src/ism_dp_kernel.h"
#include "ops/ism_dp/src/ism_dp_param.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {
at::Tensor ismDetailedPlaceForward(database::PlaceDB const& placedb,
                                   py::object               pyparam,
                                   at::Tensor               clock_available_clock_region,
                                   at::Tensor               hc_available_clock_region,
                                   at::Tensor               fixed_mask,
                                   at::Tensor               init_pos) {
  CHECK_FLAT_CPU(init_pos);
  CHECK_DIVISIBLE(init_pos, 3);
  CHECK_CONTIGUOUS(init_pos);
  ISMDetailedPlaceParam                             param = ISMDetailedPlaceParam::ParseFromPyobject(pyparam);
  auto                                              pos   = init_pos.clone();
  std::function<bool(uint32_t, uint32_t, uint32_t)> _isCKAllowedInSite;

  param.fixedMask = OPENPARF_TENSOR_DATA_PTR(fixed_mask, uint8_t);

  if (param.honorClockConstraints) {
    openparfPrint(kInfo, "Clock region constraint activated for ISM DP\n");
    _isCKAllowedInSite = [&](uint32_t clk_id, uint32_t site_x, uint32_t site_y) {
      auto hc_id = placedb.XyToHcIndex(site_x, site_y);
      auto cr_id = placedb.XyToCrIndex(site_x, site_y);
      auto a1    = clock_available_clock_region.accessor<uint8_t, 2>();
      auto a2    = hc_available_clock_region.accessor<uint8_t, 2>();
      openparfAssert(a1[clk_id][cr_id] == 0 || a1[clk_id][cr_id] == 1);
      openparfAssert(a2[clk_id][hc_id] == 0 || a2[clk_id][hc_id] == 1);
      if (a1[clk_id][cr_id] == 0 || a2[clk_id][hc_id] == 0) return false;
      return true;
    };
  } else {
    openparfPrint(kInfo, "Clock region constraint not activated for ISM DP\n");
    _isCKAllowedInSite = [&](uint32_t clk_id, uint32_t site_x, uint32_t site_y) { return true; };
  }

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "ismDetailedPlaceLauncher", [&] {
    ismDetailedPlaceLauncher<scalar_t>(placedb,
                                       param,
                                       _isCKAllowedInSite,
                                       param.honorClockConstraints,
                                       at::get_num_threads(),
                                       OPENPARF_TENSOR_DATA_PTR(pos, scalar_t));
  });
  return pos;
}

}   // namespace ism_dp

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",
        &OPENPARF_NAMESPACE::ism_dp::ismDetailedPlaceForward,
        "Independent set matching based detailed placement forward");
}
