/**
 * File              : legality_check.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.14.2020
 * Last Modified Date: 07.14.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_LEGALITY_CHECK_H
#define OPENPARF_OPS_LEGALITY_CHECK_H

#include <algorithm>
#include <vector>

// project headers
#include "container/vector_2d.hpp"
#include "database/placedb.h"
#include "geometry/box.hpp"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

using container::Vector2D;

template<typename T, typename V>
bool isClose(T x, V y, double rtol = 1e-3, double atol = 1e-6) {
  auto delta = std::abs(x - y);
  return delta <= x * rtol || delta <= atol;
}

/// Check if the 2 LUTL can be fitted into the same BLEL
/// The two LUTL IDs are for original
bool twoLUTsAreBLELCompatible(database::PlaceDB const&     db,
                              database::PlaceDB::IndexType inst1,
                              database::PlaceDB::IndexType inst2,
                              std::vector<uint8_t>&        net_markers) {
  using IndexType = database::PlaceDB::IndexType;

  // If one of the LUTL is a 6-input LUTL, they are not compatible
  if (db.isInstLUT(inst1) == 6U || db.isInstLUT(inst2) == 6U) {
    return false;
  }

  // The number of distinct input nets should be less than or equal to 5
  // Note that the _onl.pinArray() are sorted in a way that
  //   1) pins are sorted by their belonging nets from low to high
  //   2) pins belong to the same nets are adjacent

  auto markerNets = [&](IndexType inst_id, uint8_t value) {
    IndexType num_marks = 0;
    for (auto j = 0U; j < db.instPins().size2(inst_id); ++j) {
      auto pin_id = db.instPins().at(inst_id, j);
      if (db.pinSignalDirect(pin_id) == SignalDirection::kInput) {
        auto  net_id = db.pin2Net(pin_id);
        auto& marker = net_markers[net_id];
        if (marker != value) {
          marker = value;
          num_marks += 1;
        }
      }
    }
    return num_marks;
  };

  // count distinct input nets
  IndexType num_distinct_inputs = 0;
  num_distinct_inputs += markerNets(inst1, 1);
  num_distinct_inputs += markerNets(inst2, 1);

  // recover markers
  IndexType num_recover_marks = 0;
  num_recover_marks += markerNets(inst1, 0);
  num_recover_marks += markerNets(inst2, 0);

  openparfAssert(num_distinct_inputs == num_recover_marks);

  return num_distinct_inputs <= 5U;
}

template<typename T>
bool legalityCheck(database::PlaceDB const& db,
                   bool                     check_z_flag,   ///< whether check z location,
                   int                      max_clk_per_clock_region,
                   int                      max_clk_per_half_column,
                   T const*                 pos   ///< xyzxyz, assume instances align to site lower left
) {
  using IndexType             = database::PlaceDB::IndexType;

  bool        legal           = true;
  auto const& design          = db.db()->design();
  auto const& layout          = db.db()->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  auto const&         netlist      = top_module_inst->netlist();
  auto const&         site_map     = layout.siteMap();
  auto const&         resource_map = layout.resourceMap();
  auto const&         range        = db.movableRange();

  // for easy query for valid site
  Vector2D<IndexType> valid_site_map(site_map.width(), site_map.height(), std::numeric_limits<IndexType>::max());
  for (auto const& site : site_map) {
    auto const& bbox = site.bbox();
    for (IndexType ix = bbox.xl(); ix < bbox.xh(); ++ix) {
      for (IndexType iy = bbox.yl(); iy < bbox.yh(); ++iy) {
        valid_site_map(ix, iy) = site_map.index1D(site.siteMapId().x(), site.siteMapId().y());
      }
    }
  }

  auto getValidSite = [&](IndexType ix, IndexType iy) {
    auto        id1d = valid_site_map(ix, iy);
    auto const& site = site_map.at(id1d);
    openparfAssert(site);
    return site;
  };

  auto getInstName = [&](IndexType inst_id) {
    // assume flat netlist for now
    auto const& inst = netlist.inst(db.oldInstId(inst_id));
    return inst.attr().name();
  };

  // Fill nodes into the site map and check overlapping and site compatibility
  Vector2D<std::vector<IndexType>> insts_in_site_map(site_map.width(), site_map.height());
  // check each instance
  // and distribute instances to site map
  for (auto inst_id = range.first; inst_id < range.second; ++inst_id) {
    auto const& inst      = netlist.inst(db.oldInstId(inst_id));
    auto const& model     = design.model(inst.attr().modelId());
    auto        inst_x    = pos[inst_id * 3];
    auto        inst_y    = pos[inst_id * 3 + 1];
    auto        inst_z    = pos[inst_id * 3 + 2];

    auto const& site      = *getValidSite(inst_x, inst_y);
    auto const& site_type = layout.siteType(site);

    // check align to site center
    auto        center_x  = (site.bbox().xl() + site.bbox().xh()) * 0.5;
    auto        center_y  = (site.bbox().yl() + site.bbox().yh()) * 0.5;
    if (!isClose(center_x, inst_x) || !isClose(center_y, inst_y)) {
      openparfPrint(kError,
                    "inst %u (%s) @ (%g, %g, %g) failed to align to site (%u, %u) @ (%u, %u, %u, %u)\n",
                    inst_id,
                    inst.attr().name().c_str(),
                    static_cast<double>(inst_x),
                    static_cast<double>(inst_y),
                    static_cast<double>(inst_z),
                    site.siteMapId().x(),
                    site.siteMapId().y(),
                    site.bbox().xl(),
                    site.bbox().yl(),
                    site.bbox().xh(),
                    site.bbox().yh());
      legal = false;
    }

    // check site type
    auto resource_ids    = resource_map.modelResourceIds(model.id());
    bool site_type_match = false;
    for (auto resource_id : resource_ids) {
      if (site_type.resourceCapacity(resource_id)) {
        site_type_match = true;
        break;
      }
    }
    if (!site_type_match) {
      openparfPrint(kError,
                    "inst %u (%s) @ (%g, %g, %g) resource ",
                    inst_id,
                    inst.attr().name().c_str(),
                    static_cast<double>(inst_x),
                    static_cast<double>(inst_y),
                    static_cast<double>(inst_z));
      for (auto resource_id : resource_ids) {
        auto const& resource = resource_map.resource(resource_id);
        openparfPrint(kNone, "%u (%s)", resource.id(), resource.name().c_str());
      }
      openparfPrint(kNone, " failed to match site (%u, %u) [");
      for (auto resource_id = 0U; resource_id < resource_map.numResources(); ++resource_id) {
        openparfPrint(kNone, " %u", site_type.resourceCapacity(resource_id));
      }
      openparfPrint(kNone, " ]\n");
      legal = false;
    }

    insts_in_site_map(site.siteMapId().x(), site.siteMapId().y()).push_back(inst_id);
  }

  // Check clock region and half column constraints
  if (max_clk_per_clock_region != 0 || max_clk_per_half_column != 0) {
    std::vector<geometry::Box<uint32_t>>      clk_bbox(db.numClockNets());
    std::unordered_set<uint32_t>              seen_clks;
    std::vector<std::unordered_set<uint32_t>> clock_per_hc(layout.numHalfColumnRegions());
    // First, we count the clocks.
    for (auto inst_id = range.first; inst_id < range.second; ++inst_id) {
      if (db.isInstClockSource(inst_id)) continue;
      auto const& inst   = netlist.inst(db.oldInstId(inst_id));
      auto&       clocks = db.instToClocks()[inst_id];
      if (clocks.empty()) continue;
      auto inst_x = pos[inst_id * 3];
      auto inst_y = pos[inst_id * 3 + 1];
      auto inst_z = pos[inst_id * 3 + 2];
      auto cr_idx = db.XyToCrIndex(inst_x, inst_y);
      auto cr_x   = cr_idx / db.numCrY();
      auto cr_y   = cr_idx % db.numCrY();
      for (auto clk : clocks) {
        auto& b = clk_bbox[clk];
        if (seen_clks.find(clk) != seen_clks.end()) {
          if (cr_x < b.xl()) {
            b.setXL(cr_x);
          } else if (cr_x > b.xh()) {
            b.setXH(cr_x);
          }
          if (cr_y < b.yl()) {
            b.setYL(cr_y);
          } else if (cr_y > b.yh()) {
            b.setYH(cr_y);
          }
        } else {
          seen_clks.insert(clk);
          b.setXL(cr_x);
          b.setXH(cr_x);
          b.setYL(cr_y);
          b.setYH(cr_y);
        }
      }

      auto hc_idx = db.XyToHcIndex(inst_x, inst_y);
      for (auto clk : clocks) {
        clock_per_hc[hc_idx].insert(clk);
      }
    }
    std::vector<uint32_t> cr_clock_cnt(db.numCr(), 0);
    for (auto& bx : clk_bbox) {
      for (uint32_t x = bx.xl(); x <= bx.xh(); x++) {
        for (uint32_t y = bx.yl(); y <= bx.yh(); y++) {
          uint32_t cr_id = x * db.numCrY() + y;
          cr_clock_cnt[cr_id]++;
        }
      }
    }
    // Then, we check if the constraints are met
    if (max_clk_per_clock_region != 0) {
      for (uint32_t i = 0; i < cr_clock_cnt.size(); i++) {
        if (cr_clock_cnt[i] > max_clk_per_clock_region) {
          openparfPrint(kError, "Placement not legal, %i clock nets in clock region %i\n", cr_clock_cnt[i], i);
          legal = false;
        }
      }
    }
    if (max_clk_per_half_column != 0) {
      for (uint32_t i = 0; i < clock_per_hc.size(); i++) {
        if (clock_per_hc[i].size() > max_clk_per_half_column) {
          openparfPrint(kError,
                        "Placement not legal, %i clock nets in half column region %i\n",
                        clock_per_hc[i].size(),
                        i);
          legal = false;
        }
      }
    }
  }


  // check each site
  for (auto const& site : site_map) {
    auto const& site_type = layout.siteType(site);
    auto&       inst_ids  = insts_in_site_map(site.siteMapId().x(), site.siteMapId().y());
    // sort according to z location
    std::sort(inst_ids.begin(), inst_ids.end(), [&](IndexType id1, IndexType id2) {
      auto z1 = pos[id1 * 3 + 2];
      auto z2 = pos[id2 * 3 + 2];
      return z1 < z2 || (z1 == z2 && id1 < id2);
    });

    std::vector<IndexType> resource_demands(resource_map.numResources(), 0);
    // compute resource demands taken by each instance in the site
    for (auto inst_id : inst_ids) {
      auto const& inst         = netlist.inst(db.oldInstId(inst_id));
      auto const& model        = design.model(inst.attr().modelId());
      auto        resource_ids = resource_map.modelResourceIds(model.id());
      for (auto resource_id : resource_ids) {
        if (site_type.resourceCapacity(resource_id)) {
          resource_demands[resource_id] += 1;
          break;
        }
      }
    }

    // check site capacity overflow
    bool fail_flag = false;
    for (auto resource_id = 0U; resource_id < resource_map.numResources(); ++resource_id) {
      if (resource_demands[resource_id] > site_type.resourceCapacity(resource_id)) {
        fail_flag = true;
        break;
      }
    }
    if (fail_flag) {
      openparfPrint(kError,
                    "site (%u, %u) @ (%u, %u, %u, %u) [",
                    site.siteMapId().x(),
                    site.siteMapId().y(),
                    site.bbox().xl(),
                    site.bbox().yl(),
                    site.bbox().xh(),
                    site.bbox().yh());
      for (auto resource_id = 0U; resource_id < resource_map.numResources(); ++resource_id) {
        openparfPrint(kNone, " %u", site_type.resourceCapacity(resource_id));
      }
      openparfPrint(kNone, " ] resource overflow [");
      for (auto resource_id = 0U; resource_id < resource_map.numResources(); ++resource_id) {
        openparfPrint(kNone, " %u", resource_demands[resource_id]);
      }
      openparfPrint(kNone, " ] insts [");
      for (auto inst_id : inst_ids) {
        openparfPrint(kNone, " %u (%s)", inst_id, getInstName(inst_id).c_str());
      }
      openparfPrint(kNone, " ]\n");
      legal = false;
    }

    // check z location
    if (check_z_flag) {
      for (auto i = 1U; i < inst_ids.size(); ++i) {
        auto inst_id      = inst_ids[i];
        auto prev_inst_id = inst_ids[i - 1];
        auto inst_z       = pos[inst_id * 3 + 2];
        auto prev_inst_z  = pos[prev_inst_id * 3 + 2];
        if (prev_inst_z >= inst_z) {
          auto const& inst              = netlist.inst(db.oldInstId(inst_id));
          auto const& model             = design.model(inst.attr().modelId());
          auto        resource_ids      = resource_map.modelResourceIds(model.id());
          auto const& prev_inst         = netlist.inst(db.oldInstId(prev_inst_id));
          auto const& prev_model        = design.model(prev_inst.attr().modelId());
          auto        prev_resource_ids = resource_map.modelResourceIds(prev_model.id());
          std::sort(resource_ids.begin(), resource_ids.end());
          std::sort(prev_resource_ids.begin(), prev_resource_ids.end());

          // different resources, e.g., LUT/FF may have the same z locations
          if (resource_ids == prev_resource_ids) {
            openparfPrint(kError,
                          "inst %u (%s) @ (%g, %g, %g) z-overlap with inst %u (%s) @ (%g, %g, %g)\n",
                          prev_inst_id,
                          getInstName(prev_inst_id).c_str(),
                          pos[prev_inst_id * 3],
                          pos[prev_inst_id * 3 + 1],
                          pos[prev_inst_id * 3 + 2],
                          inst_id,
                          getInstName(inst_id).c_str(),
                          pos[inst_id * 3],
                          pos[inst_id * 3 + 1],
                          pos[inst_id * 3 + 2]);
            legal = false;
          }
        }
      }
    }
  }

  // check LUT compatibility
  if (check_z_flag) {
    std::vector<uint8_t> net_markers(db.numNets(), 0);
    for (auto const& site : site_map) {
      auto const& site_type       = layout.siteType(site);
      // check whether a LUT site
      IndexType   LUT_capacity    = 0;
      IndexType   FF_capacity     = 0;
      IndexType   LUTRAM_capacity = 0;
      for (IndexType resource_id = 0; resource_id < resource_map.numResources(); ++resource_id) {
        auto capacity = site_type.resourceCapacity(resource_id);
        if (capacity) {
          auto category = db.resourceCategory(resource_id);
          switch (category) {
            case ResourceCategory::kLUTL:
              LUT_capacity += capacity;
              break;
            case ResourceCategory::kLUTM:
              LUT_capacity += capacity;
              LUTRAM_capacity = 1;
              break;
            case ResourceCategory::kFF:
              FF_capacity += capacity;
              break;
            default:
              break;
          }
        }
      }
      std::vector<IndexType> LUT_ids(LUT_capacity, std::numeric_limits<IndexType>::max());
      std::vector<IndexType> FF_ids(FF_capacity, std::numeric_limits<IndexType>::max());
      auto const&            inst_ids = insts_in_site_map(site.siteMapId().x(), site.siteMapId().y());
      for (auto inst_id : inst_ids) {
        auto inst_z = pos[inst_id * 3 + 2];
        if (db.isInstLUT(inst_id)) {
          openparfAssert(inst_z < LUT_capacity);
          LUT_ids[IndexType(inst_z)] = inst_id;
        } else if (db.isInstFF(inst_id)) {
          openparfAssert(inst_z < FF_capacity);
          FF_ids[IndexType(inst_z)] = inst_id;
        }
      }
      for (IndexType i = 0; i < LUT_capacity; i += 2) {
        auto lut1 = LUT_ids[i];
        auto lut2 = LUT_ids[i + 1];
        // Make sure the two LUTs are compatible
        if (lut1 != std::numeric_limits<uint32_t>::max() && lut2 != std::numeric_limits<uint32_t>::max()) {
          if (!twoLUTsAreBLELCompatible(db, lut1, lut2, net_markers)) {
            openparfPrint(kError,
                          "LUT %u (%s) @ (%g, %g, %g) and LUT %u (%s) @ (%g, %g, %g) are not compatible\n",
                          lut1,
                          getInstName(lut1).c_str(),
                          pos[lut1 * 3],
                          pos[lut1 * 3 + 1],
                          pos[lut1 * 3 + 2],
                          lut2,
                          getInstName(lut2).c_str(),
                          pos[lut2 * 3],
                          pos[lut2 * 3 + 1],
                          pos[lut2 * 3 + 2]);
            legal = false;
          }
        }
        // Make sure LUT6 are at odd position
        if (lut1 != std::numeric_limits<uint32_t>::max()) {
          if (db.isInstLUT(lut1) == 6U) {
            openparfPrint(kError,
                          "LUT %u (%s) @ (%g, %g, %g) is placed at even z-location\n",
                          lut1,
                          getInstName(lut1).c_str(),
                          pos[lut1 * 3],
                          pos[lut1 * 3 + 1],
                          pos[lut1 * 3 + 2]);
            legal = false;
          }
        }
      }


      // check SHIFT/Distributed RAM
      if (LUTRAM_capacity > 0) {
        // SLICEM
        int32_t              LRAM_counter = 0;
        std::vector<int32_t> lram_inst_ids;
        std::vector<int32_t> ff_masked_slot(16, 0);
        std::vector<int32_t> lram_slot(16, 0);
        for (auto inst_id : inst_ids) {
          auto const& inst  = netlist.inst(db.oldInstId(inst_id));
          auto const& model = design.model(inst.attr().modelId());
          if (model.name() == "LRAM" || model.name() == "SHIFT") {
            LRAM_counter++;
            lram_inst_ids.push_back(inst_id);
          }
        }
        if (LRAM_counter > 0) {
          if (LRAM_counter > 8) {
            legal = false;
            openparfPrint(kError,
                          "More than 8 SHIFT/LUTRAM @ site (%u, %u): \n",
                          site.siteMapId().x(),
                          site.siteMapId().y());

            for (auto inst_id : inst_ids) {
              auto const& inst  = netlist.inst(db.oldInstId(inst_id));
              auto const& model = design.model(inst.attr().modelId());
              int32_t     loc_z = pos[inst_id * 3 + 2];
              if (model.name() == "LRAM" || model.name() == "SHIFT") {
                // loc_z of LRAM/SHIFT: 1 3 5 7 9 11 13 15
                openparfPrint(kError,
                              "    %s %u(%s) @ (%g, %g, %g) \n",
                              model.name().c_str(),
                              inst_id,
                              inst.attr().name().c_str(),
                              pos[inst_id * 3],
                              pos[inst_id * 3 + 1],
                              pos[inst_id * 3 + 2]);
              }
            }
          }

          for (auto inst_id : inst_ids) {
            auto const& inst  = netlist.inst(db.oldInstId(inst_id));
            auto const& model = design.model(inst.attr().modelId());
            int32_t     loc_z = pos[inst_id * 3 + 2];
            if (model.name() == "LRAM" || model.name() == "SHIFT") {
              // loc_z of LRAM/SHIFT: 1 3 5 7 9 11 13 15
              ff_masked_slot[loc_z]     = 1;
              ff_masked_slot[loc_z ^ 1] = 1;
              lram_slot[loc_z]          = inst_id;
              lram_slot[loc_z ^ 1]      = inst_id;
            }
          }

          for (auto inst_id : inst_ids) {
            auto const& inst  = netlist.inst(db.oldInstId(inst_id));
            auto const& model = design.model(inst.attr().modelId());
            int32_t     loc_z = pos[inst_id * 3 + 2];
            if (db.isInstLUT(inst_id) && model.name() != "LRAM" && model.name() != "SHIFT") {
              int32_t     lram_inst_id = lram_inst_ids[0];
              auto const& lram_inst    = netlist.inst(db.oldInstId(lram_inst_id));
              auto const& lram_model   = design.model(lram_inst.attr().modelId());
              legal                    = false;
              openparfPrint(kError,
                            "%s %u(%s) @ (%g, %g, %g) overlaps with %s %u(%s) @(%g, %g, %g)\n",
                            model.name().c_str(),
                            inst_id,
                            inst.attr().name().c_str(),
                            pos[inst_id * 3],
                            pos[inst_id * 3 + 1],
                            pos[inst_id * 3 + 2],
                            lram_model.name().c_str(),
                            lram_inst_id,
                            lram_inst.attr().name().c_str(),
                            pos[lram_inst_id * 3],
                            pos[lram_inst_id * 3 + 1],
                            pos[lram_inst_id * 3 + 2]);
            }
            if (db.isInstFF(inst_id) && ff_masked_slot[loc_z] == 1) {
              int32_t     lram_inst_id = lram_slot[loc_z];
              auto const& lram_inst    = netlist.inst(db.oldInstId(lram_inst_id));
              auto const& lram_model   = design.model(lram_inst.attr().modelId());
              legal                    = false;
              openparfPrint(kError,
                            "%s %u(%s) @ (%g, %g, %g) overlaps with %s %u(%s) @(%g, %g, %g)\n",
                            model.name().c_str(),
                            inst_id,
                            inst.attr().name().c_str(),
                            pos[inst_id * 3],
                            pos[inst_id * 3 + 1],
                            pos[inst_id * 3 + 2],
                            lram_model.name().c_str(),
                            lram_inst_id,
                            lram_inst.attr().name().c_str(),
                            pos[lram_inst_id * 3],
                            pos[lram_inst_id * 3 + 1],
                            pos[lram_inst_id * 3 + 2]);
            }
          }
        }
      } else {
        // SLICEL
        int32_t LRAM_counter = 0;
        for (auto inst_id : inst_ids) {
          auto const& inst  = netlist.inst(db.oldInstId(inst_id));
          auto const& model = design.model(inst.attr().modelId());
          if (model.name() == "LRAM" || model.name() == "SHIFT") {
            LRAM_counter++;
          }
        }
        if (LRAM_counter > 0) {
          legal = false;
          openparfPrint(kError,
                        "SHIFT/LUTRAM cannot be placed in SLICEL site (%u, %u): \n",
                        site.siteMapId().x(),
                        site.siteMapId().y());

          for (auto inst_id : inst_ids) {
            auto const& inst  = netlist.inst(db.oldInstId(inst_id));
            auto const& model = design.model(inst.attr().modelId());
            if (model.name() == "LRAM" || model.name() == "SHIFT") {
              openparfPrint(kError,
                            "    %s %u(%s) @ (%g, %g, %g) \n",
                            model.name().c_str(),
                            inst_id,
                            inst.attr().name().c_str(),
                            pos[inst_id * 3],
                            pos[inst_id * 3 + 1],
                            pos[inst_id * 3 + 2]);
            }
          }
        }
      }

      // Check FF control sets
      for (auto fi = LUT_capacity; fi < FF_ids.size(); fi += FF_capacity / db.numControlSetsPerCLB()) {
        // net of clock, SR, and CE signals
        IndexType                ck = std::numeric_limits<IndexType>::max(), sr = std::numeric_limits<IndexType>::max();
        std::array<IndexType, 2> ce = {std::numeric_limits<IndexType>::max(), std::numeric_limits<IndexType>::max()};
        for (auto i = 0U; i < FF_capacity / db.numControlSetsPerCLB(); ++i) {
          auto ff = FF_ids[fi + i];
          if (ff != std::numeric_limits<IndexType>::max()) {
            IndexType k = i % 2;
            for (auto j = 0U; j < db.instPins().size2(ff); ++j) {
              auto pin_id = db.instPins().at(ff, j);
              auto net_id = db.pin2Net(pin_id);
              switch (db.pinSignalType(pin_id)) {
                case SignalType::kClock:
                  if (ck == std::numeric_limits<IndexType>::max()) {
                    ck = net_id;
                  } else if (ck != net_id) {
                    openparfPrint(kError,
                                  "FF %u (%s) @ (%g, %g, %g) invalid Clock signal pin %u, net %u, target net %u\n",
                                  ff,
                                  getInstName(ff).c_str(),
                                  pos[ff * 3],
                                  pos[ff * 3 + 1],
                                  pos[ff * 3 + 2],
                                  pin_id,
                                  net_id,
                                  ck);
                    legal = false;
                  }
                  break;
                case SignalType::kControlSR:
                  if (sr == std::numeric_limits<IndexType>::max()) {
                    sr = net_id;
                  } else if (sr != net_id) {
                    openparfPrint(kError,
                                  "FF %u (%s) @ (%g, %g, %g) invalid ControlSR signal pin %u, net %u, target net %u\n",
                                  ff,
                                  getInstName(ff).c_str(),
                                  pos[ff * 3],
                                  pos[ff * 3 + 1],
                                  pos[ff * 3 + 2],
                                  pin_id,
                                  net_id,
                                  sr);
                    legal = false;
                  }
                  break;
                case SignalType::kControlCE:
                  if (ce[k] == std::numeric_limits<IndexType>::max()) {
                    ce[k] = net_id;
                  } else if (ce[k] != net_id) {
                    openparfPrint(
                            kError,
                            "FF %u (%s) @ (%g, %g, %g) invalid ControlCE[%u] signal pin %u, net %u, target net %u\n",
                            ff,
                            getInstName(ff).c_str(),
                            pos[ff * 3],
                            pos[ff * 3 + 1],
                            pos[ff * 3 + 2],
                            k,
                            pin_id,
                            net_id,
                            ce[k]);
                    legal = false;
                  }
                  break;
                default:
                  break;
              }
            }
          }
        }
      }
    }
  }

  return legal;
}

template<class T>
bool xarchLegalityCheck(database::PlaceDB const& db,
                     int32_t                  num_cc,
                     int32_t                  num_sc,
                     int32_t*                 cla_bs,
                     int32_t*                 cla_starts,
                     int32_t*                 lut_bs,
                     int32_t*                 lut_starts,
                     int32_t*                 ssr_bs,
                     int32_t*                 ssr_starts,
                     T*                       pos) {
  using IndexType             = database::PlaceDB::IndexType;

  bool        legal           = true;
  auto const& design          = db.db()->design();
  auto const& layout          = db.db()->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  auto const& netlist     = top_module_inst->netlist();

  auto        getInstName = [&](IndexType inst_id) {
    // assume flat netlist for now
    auto const& inst = netlist.inst(db.oldInstId(inst_id));
    return inst.attr().name();
  };

  // check CLA & LUT alignment
  for (int32_t cc = 0; cc < num_cc; cc++) {
    int32_t cla_beg = cla_starts[cc];
    int32_t cla_end = cla_starts[cc + 1];
    int32_t lut_beg = lut_starts[cc];
    T       gx      = pos[cla_bs[cla_beg] * 3];
    T       gy      = pos[cla_bs[cla_beg] * 3 + 1];
    int32_t gz      = pos[cla_bs[cla_beg] * 3 + 2];
    for (int32_t cla_pt = cla_beg; cla_pt < cla_end; cla_pt++) {
      int32_t cla = cla_bs[cla_pt];
      T       cx  = pos[cla * 3];
      T       cy  = pos[cla * 3 + 1];
      T       cz  = pos[cla * 3 + 2];
      if (gx != cx || gy != cy || gz != cz) {
        legal = false;
        openparfPrint(kError, "CLA %u (%s) @ (%g, %g, %g) is not aligned\n", cla, getInstName(cla).c_str(), cx, cy, cz);
      }
      for (int32_t k = 0; k < 4; k++) {
        int32_t lut = lut_bs[lut_beg + (cla_pt - cla_beg) * 4 + k];
        T       lx  = pos[lut * 3];
        T       ly  = pos[lut * 3 + 1];
        T       lz  = pos[lut * 3 + 2];
        if (lx != gx || ly != gy || lz != gz * 8 + k * 2 + 1) {
          legal = false;
          openparfPrint(kError,
                        "LUT %u (%s) @ (%g, %g, %g) is not aligned with dependent CLA %u (%s) @ (%g, %g, %g)\n",
                        lut,
                        getInstName(lut).c_str(),
                        lx,
                        ly,
                        lz,
                        cla,
                        getInstName(cla).c_str(),
                        cx,
                        cy,
                        cz);
        }
      }
      gz += 1;
      if (gz >= 2) {
        gz = 0;
        gy++;
      }
    }
  }

  // check GCU0 alignment
  for (int32_t sc = 0; sc < num_sc; sc++) {
    int32_t ssr_beg = ssr_starts[sc];
    int32_t ssr_end = ssr_starts[sc + 1];
    T       gx      = pos[ssr_bs[ssr_beg] * 3];
    T       gy      = pos[ssr_bs[ssr_beg] * 3 + 1];
    for (int32_t ssr_pt = ssr_beg; ssr_pt < ssr_end; ssr_pt++) {
      int32_t gcu = ssr_bs[ssr_pt];
      T       cx  = pos[gcu * 3];
      T       cy  = pos[gcu * 3 + 1];
      T       cz  = pos[gcu * 3 + 2];
      if (gx != cx || gy != cy || cz != 0) {
        legal = false;
        openparfPrint(kError,
                      "GCU0 %u (%s) @ (%g, %g, %g) is not aligned\n",
                      gcu,
                      getInstName(gcu).c_str(),
                      cx,
                      cy,
                      cz);
      }
      gy += 4;
    }
  }
  return legal;
}
OPENPARF_END_NAMESPACE

#endif
