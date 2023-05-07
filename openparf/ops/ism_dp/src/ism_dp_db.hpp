/**
 * File              : ism_dp_db.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.02.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_ISM_DP_ISM_DP_DB_HPP
#define OPENPARF_OPS_ISM_DP_ISM_DP_DB_HPP

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {
template<typename T>
struct ISMDetailedPlaceDB {
  int32_t const*  inst_pins                = nullptr;   ///< #pins
  int32_t const*  inst_pins_begin          = nullptr;   ///< #insts + 1
  int32_t const*  net_pins                 = nullptr;   ///< #pins
  int32_t const*  net_pins_begin           = nullptr;   ///< #nets + 1
  int32_t const*  pin2inst                 = nullptr;   ///< #pins
  int32_t const*  pin2net                  = nullptr;   ///< #pins

  uint8_t const*  pin_signal_directs       = nullptr;   ///< #pins
  uint8_t const*  pin_signal_types         = nullptr;   ///< #pins
  uint32_t const* inst_models              = nullptr;   ///< #insts
  uint8_t const*  inst_resource_categories = nullptr;   ///< #insts
  int32_t const*  inst_CLOCK_pins          = nullptr;   ///< #insts, clock pin indices, infinity if not exists
  int32_t const*  inst_control_SR_pins     = nullptr;   ///< #insts, control SR pin indices, infinity if not exists
  int32_t const*  inst_control_CE_pins     = nullptr;   ///< #insts, control CE pin indices, infinity if not exists
  T const*        inst_sizes               = nullptr;   ///< #insts x 2
  T const*        site_bboxes              = nullptr;   ///< #sites x 4
  T const*        net_weights              = nullptr;   ///< #nets
  int32_t const*  site_capacities          = nullptr;   ///< #sites * #resources
  int32_t const*  site_LUT_capacities      = nullptr;   ///< #sites
  int32_t const*  site_FF_capacities       = nullptr;   ///< #sites
  int32_t const*  site_DSP_capacities      = nullptr;   ///< #sites
  int32_t const*  site_RAM_capacities      = nullptr;   ///< #sites
  int32_t const*  resource_categories      = nullptr;

  T*              pos                      = nullptr;        ///< #insts * 3, location of instances
  T*              pos_xy                   = nullptr;        ///< #insts * 2, locations of instances
  T*              pin_pos                  = nullptr;        ///< #pins * 2, pin locations
  T*              rudy_map[2]        = {nullptr, nullptr};   ///< bin_map_dim * 2, horizontal and vertical RUDY map
  T*              pin_util_map       = nullptr;              ///< bin_map_dim, pin utilization map
  T*              inst_num_pins      = nullptr;              ///< #insts, number of pins per inst
  T*              stretch_inst_sizes = nullptr;              ///< #insts * 2, stretched inst sizes for pin utilization

  int32_t         num_insts;         ///< number of instances
  int32_t         num_nets;          ///< number of nets
  int32_t         num_pins;          ///< number of pins
  int32_t         num_sites;         ///< number of sites
  int32_t         num_resources;     ///< number of resources
  int32_t         site_map_dim[2];   ///< dimension of site map
  T               xl, yl, xh, yh;
};

}   // namespace ism_dp
OPENPARF_END_NAMESPACE

#endif
