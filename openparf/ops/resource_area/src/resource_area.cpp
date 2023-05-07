/**
 * @file   resource_area.cpp
 * @author Yibai Meng
 * @date   Aug 2020
 */

#include "ops/resource_area/src/resource_area_kernel.hpp"
#include "util/torch.h"
#include "util/util.h"


OPENPARF_BEGIN_NAMESPACE

at::Tensor computeResourceAreas(at::Tensor pos,
        at::Tensor                         is_luts,
        at::Tensor                         is_ffs,
        at::Tensor                         ff_ctrlsets,
        int32_t                            cksr_size,
        int32_t                            ce_size,
        int32_t                            num_bins_x,
        int32_t                            num_bins_y,
        double                             stddev_x,
        double                             stddev_y,
        double                             stddev_trunc,
        int32_t                            slice_capacity,
        const std::string                 &packing_rule_name) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(is_luts);
  CHECK_CONTIGUOUS(is_luts);
  CHECK_FLAT_CPU(is_ffs);
  CHECK_CONTIGUOUS(is_ffs);
  CHECK_CONTIGUOUS(ff_ctrlsets);

  PackingRules packing_rule;
  if (packing_rule_name.compare("ultrascale") == 0) {
    packing_rule = PackingRules::kUltraScale;
  } else if (packing_rule_name.compare("xarch") == 0) {
    packing_rule = PackingRules::kXarch;
  } else {
    openparfPrint(kError, "Don't support packing rule `%s`\n", packing_rule_name.c_str());
  }

  // TODO(Yibai Meng): don't hardcode the 6
  at::Tensor lut_demand_map = at::zeros({num_bins_x, num_bins_y, 6}, pos.options());
  at::Tensor ff_demand_map  = at::zeros({num_bins_x, num_bins_y, cksr_size, ce_size}, pos.options());
  // First, we compute the demand map: that is, the amount of each type of lut need
  // by the current placment(pos) at each location(bins)
  // Each instance doesn't only affect the bins it overlaps. It affects
  int32_t    num_insts      = is_luts.sizes()[0];   // Number of movable and fixed insts.
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "fillGaussianDemandMapLauncher", [&] {
    fillGaussianDemandMapLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(is_luts, uint8_t), OPENPARF_TENSOR_DATA_PTR(is_ffs, uint8_t),
            OPENPARF_TENSOR_DATA_PTR(ff_ctrlsets, int32_t), num_bins_x, num_bins_y, num_insts, cksr_size, ce_size,
            stddev_x, stddev_y, stddev_trunc, at::get_num_threads(), OPENPARF_TENSOR_DATA_PTR(lut_demand_map, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(ff_demand_map, scalar_t));
  });

  // Now, we calculate the inflated area per bin
  at::Tensor lut_area_map = at::zeros({num_bins_x, num_bins_y, 6}, pos.options());
  at::Tensor ff_area_map  = at::zeros({num_bins_x, num_bins_y, cksr_size, ce_size}, pos.options());
  CHECK_FLAT_CPU(lut_area_map);
  CHECK_CONTIGUOUS(lut_area_map);
  CHECK_FLAT_CPU(ff_area_map);
  CHECK_CONTIGUOUS(ff_area_map);
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "computeInstanceAreaMapLauncher", [&] {
    computeInstanceAreaMapLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(lut_demand_map, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(ff_demand_map, scalar_t), num_bins_x, num_bins_y, stddev_x, stddev_y, stddev_trunc,
            cksr_size, ce_size, slice_capacity, at::get_num_threads(), OPENPARF_TENSOR_DATA_PTR(lut_area_map, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(ff_area_map, scalar_t), packing_rule);
  });

  // Finally, we get the inflated area attributed to each instance:
  at::Tensor inst_areas = at::zeros(is_luts.sizes()[0], pos.options());
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "collectInstanceAreasLauncher", [&] {
    collectInstanceAreasLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(lut_area_map, scalar_t), OPENPARF_TENSOR_DATA_PTR(ff_area_map, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(is_luts, uint8_t), OPENPARF_TENSOR_DATA_PTR(is_ffs, uint8_t),
            OPENPARF_TENSOR_DATA_PTR(ff_ctrlsets, int32_t), num_bins_x, num_bins_y, is_luts.sizes()[0], cksr_size,
            ce_size, slice_capacity, stddev_x, stddev_y, stddev_trunc, at::get_num_threads(),
            OPENPARF_TENSOR_DATA_PTR(inst_areas, scalar_t));
  });

  return inst_areas;
}

std::tuple<at::Tensor, int32_t, int32_t> computeControlSets(at::Tensor inst_pins,
        at::Tensor                                                     inst_pins_start,
        at::Tensor                                                     pin2net_map,
        at::Tensor                                                     is_inst_FFs,
        at::Tensor                                                     pin_signal_types) {
  int32_t num_insts   = inst_pins_start.numel() - 1;
  auto    ff_ctrlsets = at::zeros({num_insts, 2}, inst_pins.options());
  int32_t num_cksr    = 0;
  int32_t num_ce      = 0;

  computeControlSetsLauncher(OPENPARF_TENSOR_DATA_PTR(inst_pins, int32_t),
          OPENPARF_TENSOR_DATA_PTR(inst_pins_start, int32_t), OPENPARF_TENSOR_DATA_PTR(pin2net_map, int32_t),
          OPENPARF_TENSOR_DATA_PTR(is_inst_FFs, uint8_t), OPENPARF_TENSOR_DATA_PTR(pin_signal_types, uint8_t),
          num_insts, OPENPARF_TENSOR_DATA_PTR(ff_ctrlsets, int32_t), num_cksr, num_ce);

  return {ff_ctrlsets, num_cksr, num_ce};
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_resource_areas", &OPENPARF_NAMESPACE::computeResourceAreas,
          "compute the resource area for LUTs and ffs");
  m.def("compute_control_sets", &OPENPARF_NAMESPACE::computeControlSets, "compute control sets for FFs");
}
