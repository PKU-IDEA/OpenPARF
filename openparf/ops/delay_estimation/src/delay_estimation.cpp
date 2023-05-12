#include "delay_estimation.h"

// c++ library headers
#include <algorithm>
#include <limits>
#include <tuple>

OPENPARF_BEGIN_NAMESPACE

namespace delay_estimation {

/**
 * @brief
 *
 * @tparam T
 * @param placedb
 * @param pos
 * @param net_mask_ignore_large
 * @param[out] pin_feature
 * @param[out] ignored_pin_masks
 */
template<class T>
void ExtractDelayFeatures(database::PlaceDB const& placedb,
                          T*                       pos,
                          uint8_t*                 net_mask_ignore_large,
                          T*                       pin_feature,
                          bool*                    ignored_pin_masks) {
  auto    db                              = placedb.db();
  auto&   design                          = db->design();
  auto&   layout                          = db->layout();
  auto    top_module_inst                 = design.topModuleInst();
  auto&   netlist                         = top_module_inst->netlist();
  auto&   site_map                        = layout.siteMap();
  int32_t vdd_vss_net_id                  = design.VddVssNetId();
  int32_t num_pins                        = netlist.numPins();
  int32_t site_map_width                  = site_map.width();
  int32_t feature_size                    = site_map_width + 3;

  // ignore clock net & VDD/VSS net & larget net
  auto    isClockNetOrVddVssNetorLargeNet = [&placedb, &vdd_vss_net_id, &net_mask_ignore_large](int32_t net_id) {
    return (placedb.netIdToClockId(net_id) != InvalidIndex<database::PlaceDB::IndexType>::value ||
            net_id == vdd_vss_net_id || net_mask_ignore_large[net_id] == 0);
  };

  for (auto const& net_id : netlist.netIds()) {
    const auto& net     = netlist.net(net_id);
    const auto& pin_ids = net.pinIds();
    if (isClockNetOrVddVssNetorLargeNet(net_id)) {
      // ignore clock net & VDD/VSS net & larget net
      for (const auto& pin_id : pin_ids) {
        ignored_pin_masks[pin_id] = true;
      }
      continue;
    }

    int32_t net_size      = pin_ids.size();
    int32_t driver_pin_id = std::numeric_limits<int32_t>::max();
    int32_t source_inst_id;
    T       driver_x;
    T       driver_y;

    for (const auto& pin_id : pin_ids) {
      const auto& pin = netlist.pin(pin_id);
      if (pin.attr().signalDirect() == SignalDirection::kOutput) {
        openparfAssert(driver_pin_id == std::numeric_limits<int32_t>::max());
        driver_pin_id  = pin_id;
        source_inst_id = pin.instId();
        driver_x       = pos[source_inst_id << 1];
        driver_y       = pos[source_inst_id << 1 | 1];
      }
    }
    openparfAssert(driver_pin_id != std::numeric_limits<int32_t>::max());
    ignored_pin_masks[driver_pin_id] = true;

    for (const auto& pin_id : pin_ids) {
      if (pin_id != driver_pin_id) {
        T*          feature        = pin_feature + feature_size * pin_id;
        const auto& pin            = netlist.pin(pin_id);
        int32_t     target_inst_id = pin.instId();
        T           sink_x         = pos[target_inst_id << 1];
        T           sink_y         = pos[target_inst_id << 1 | 1];
        T           min_x          = std::min(driver_x, sink_x);
        T           max_x          = std::max(driver_x, sink_x);
        int32_t     lb             = static_cast<int32_t>(std::ceil(min_x));
        int32_t     rb             = static_cast<int32_t>(std::floor(max_x));

        // column covering features
        if (lb > 0) feature[lb - 1] = lb - min_x;
        if (rb < site_map_width) feature[rb] = max_x - rb;
        for (int32_t i = lb; i < rb; i++) feature[i] = 1;
        // displacement in the y direction
        feature[site_map_width] = std::fabs(driver_y - sink_y);
        // whether the driver/sink is RAM
        if (placedb.isInstRAM(source_inst_id) || placedb.isInstRAM(target_inst_id)) feature[site_map_width + 1] = 1;
        // whether the driver/sink is DSP
        if (placedb.isInstDSP(source_inst_id) || placedb.isInstDSP(target_inst_id)) feature[site_map_width + 2] = 1;
      }
    }
  }
}

/**
 * @brief Construct the output pin featues. The feature size is `the layout width` + 1 (displacement in the y direction
 * + 1 (whether the driver/sink is BRAM) + 1 (whether the driver/sink is DSP).
 * We do not consider the delay of three kinds of pins:
 *  1. The pins of clock nets
 *  2. The pins of VDD/VSS net
 *  3. The driver pins
 * The features of these pins are useless, and are set to zero tensors by default.
 *
 * @param placedb
 * @param pos
 * @return at::Tensor pin_features
 */
std::tuple<at::Tensor, at::Tensor> DelayEstimationForward(database::PlaceDB const& placedb,
                                                          at::Tensor               net_mask_ignore_large,
                                                          at::Tensor               pos) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(net_mask_ignore_large);
  CHECK_CONTIGUOUS(net_mask_ignore_large);

  auto       db               = placedb.db();
  auto&      design           = db->design();
  auto&      layout           = db->layout();
  auto       top_module_inst  = design.topModuleInst();
  auto&      netlist          = top_module_inst->netlist();
  auto&      site_map         = layout.siteMap();
  int32_t    num_pins         = netlist.numPins();
  int32_t    site_map_width   = site_map.width();
  int32_t    feature_size     = site_map_width + 3;
  at::Tensor pin_features     = at::zeros({num_pins, feature_size}, pos.options());
  at::Tensor ignore_pin_masks = at::zeros({num_pins}, pos.options()).to(torch::kBool);
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "ExtractDelayFeatures", [&] {
    ExtractDelayFeatures<scalar_t>(placedb,
                                   OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                   OPENPARF_TENSOR_DATA_PTR(net_mask_ignore_large, uint8_t),
                                   OPENPARF_TENSOR_DATA_PTR(pin_features, scalar_t),
                                   OPENPARF_TENSOR_DATA_PTR(ignore_pin_masks, bool));
  });
  return {pin_features, ignore_pin_masks};
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void ExtractDelayFeatures<T>(database::PlaceDB const& placedb,                                              \
                                        T*                       pos,                                                  \
                                        uint8_t*                 net_mask_ignore_large,                                \
                                        T*                       pin_features,                                         \
                                        bool*                    ignore_pin_masks);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace delay_estimation

OPENPARF_END_NAMESPACE
