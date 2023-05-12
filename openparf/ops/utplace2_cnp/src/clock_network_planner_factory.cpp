#include "clock_network_planner_factory.h"

#include <functional>

#include "assignment_group_factory.h"
#include "clock_demand_estimation_factory.h"
#include "clock_info_group.h"
#include "clock_region_assignment_result_factory.h"
#include "cr_id2ck_sink_set_factory.h"
#include "cr_id_rsrc_id2site_map.h"
#include "cr_id_rsrc_id2site_map_factory.h"
#include "database/clock_region.h"
#include "max_flow_graph_factory.h"
#include "min_cost_flow_graph_factory.h"
#include "util/enums.h"
#include "util/message.h"

namespace clock_network_planner {

using namespace openparf;
using openparf::ResourceCategory;
using openparf::database::ClockRegion;
using openparf::database::ClockRegionMap;
using openparf::database::Layout;
using openparf::database::PlaceDB;
using openparf::database::Site;
using openparf::geometry::Point;

using IndexType   = ClockNetworkPlannerImpl::IndexType;
using RealType    = AssignmentGroupImpl::RealType;
using FlowIntType = ClockNetworkPlannerImpl::FlowIntType;

void ClockNetworkPlannerFactory::commitResult(ClockNetworkPlanner                         cnp,
                                              ClockRegionAssignmentResultWrapper          region_assignment_wrapper,
                                              ClockNetworkPlannerImpl::ClockAvailCRArray &ck_avail_crs,
                                              PlaceDB &                                   db) {
  IndexType                   x_cr_num = cnp.x_crs_num();
  IndexType                   y_cr_num = cnp.y_crs_num();
  std::vector<Box<IndexType>> ck_avail_cr_boxes(cnp.cks_num());
  for (auto &box : ck_avail_cr_boxes) box.reset();
  for (IndexType cr_id = 0; cr_id < cnp.crs_num(); cr_id++) {
    IndexType      x_cr_id = cr_id / y_cr_num;
    IndexType      y_cr_id = cr_id % y_cr_num;
    Box<IndexType> cr_box(x_cr_id, y_cr_id, x_cr_id, y_cr_id);
    for (auto &inst_id : region_assignment_wrapper.cr_to_all_inst_array()[cr_id]) {
      if (!db.isInstClockSource(inst_id)) {
        // Ignore the clock source when computing the clock region bounding box.
        for (auto &ck_id : db.instToClocks()[inst_id]) {
          ck_avail_cr_boxes[ck_id].join(cr_box);
        }
      }
    }
  }
  ck_avail_crs.resize(cnp.cks_num());
  using IndexSet = std::set<IndexType>;
  using openparf::container::Vector2D;
  Vector2D<IndexSet> ck_demand_grid;
  ck_demand_grid.resize(x_cr_num, y_cr_num);
  for (IndexType ck_id = 0; ck_id < cnp.cks_num(); ck_id++) {
    ck_avail_crs.at(ck_id).resize(cnp.crs_num());
    std::fill(ck_avail_crs.at(ck_id).begin(), ck_avail_crs.at(ck_id).end(), 0);
    Box<IndexType> cr_box = ck_avail_cr_boxes.at(ck_id);
    for (IndexType x = cr_box.xl(); x <= cr_box.xh(); x++) {
      for (IndexType y = cr_box.yl(); y <= cr_box.yh(); y++) {
        IndexType cr_id                  = x * y_cr_num + y;
        ck_avail_crs.at(ck_id).at(cr_id) = 1;
        ck_demand_grid.at(x, y).insert(ck_id);
      }
    }
  }
  openparfPrint(kDebug, "Final Clock Demand Grid:\n");
  for (int x_cr_id = 0; x_cr_id < x_cr_num; x_cr_id++) {
    std::stringstream ss;
    for (int y_cr_id = 0; y_cr_id < y_cr_num; y_cr_id++) {
      ss << std::setfill(' ') << std::setw(2) << ck_demand_grid.at(x_cr_id, y_cr_id).size() << " ";
    }
    openparfPrint(kDebug, "%s\n", std::string(ss.str()).c_str());
  }
  //    for (int ck_id = 0; ck_id < cnp.cks_num(); ck_id++) {
  //        auto &box = ck_avail_cr_boxes.at(ck_id);
  //        openparfPrint(kDebug, "%d: xl = %d, yl = %d, xh = %d, yh = %d\n", ck_id, box.xl(), box.yl(),
  //                      box.xh(), box.yh());
  //    }
}

ClockNetworkPlanner ClockNetworkPlannerFactory::genClockNetworkPlannerFromPlaceDB(PlaceDB &db) {
  using openparf::database::ClockRegion;

  auto                impl = std::make_shared<ClockNetworkPlannerImpl>();
  ClockNetworkPlanner cnp(impl);

  {
    impl->x_cnp_bin_size_                        = db.cnpBinSizes().first;
    impl->y_cnp_bin_size_                        = db.cnpBinSizes().second;
    impl->x_crs_num_                             = db.numCrX();
    impl->y_crs_num_                             = db.numCrY();
    impl->crs_num_                               = db.numCr();
    impl->cks_num_                               = db.numClockNets();
    impl->insts_num_                             = db.numInsts();
    impl->clock_region_capacity_                 = db.ClockRegionCapacity();
    impl->max_num_clock_pruning_per_iteration_   = db.cnpUtplace2MaxNumClockPruningPerIteration();
    impl->enable_packing_                        = db.cnpUtplace2EnablePacking();
    impl->flow_size_per_packed_area_dem_         = 16;
    impl->min_cost_max_flow_cost_scaling_factor_ = 1000.0;
  }

  initializeBins(db, impl->bin_boxes_array_, impl->bin_id_size_map_);

  /* get resource ids for the packed group and single-element groups. */ {
    auto &packed_resource_ids           = impl->packed_resource_ids_;
    auto &single_ele_group_resource_ids = impl->single_ele_group_resource_ids_;
    packed_resource_ids.clear();
    single_ele_group_resource_ids.clear();

    IndexType resource_num = db.numResources();
    for (IndexType resource_id = 0; resource_id < resource_num; resource_id++) {
      auto resource_category = db.resourceCategory(resource_id);
      switch (resource_category) {
        case ResourceCategory::kLUTL:
        case ResourceCategory::kLUTM:
        case ResourceCategory::kFF:
          packed_resource_ids.emplace_back(resource_id);
          break;
        case ResourceCategory::kSSSIR:
          single_ele_group_resource_ids.emplace_back(resource_id);
          break;
        default:
          continue;
      }
    }
  }

  namespace arg = std::placeholders;

  /* Binding functions for generating group assignment. */ {
    impl->genGroupArrayWrapperFloat_ = [cnp, &db](auto &&pos, auto &&inst_areas) {
      return AssignmentGroupFactory::genGroupArrayWrapperFromPlaceDBTemplate<float>(
              cnp,
              db,
              std::forward<decltype(pos)>(pos),
              std::forward<decltype(inst_areas)>(inst_areas));
    };
    impl->genGroupArrayWrapperDouble_ = [cnp, &db](auto &&pos, auto &&inst_areas) {
      return AssignmentGroupFactory::genGroupArrayWrapperFromPlaceDBTemplate<double>(
              cnp,
              db,
              std::forward<decltype(pos)>(pos),
              std::forward<decltype(inst_areas)>(inst_areas));
    };
  }

  CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map = CrIdRsrcId2SiteMapFactory::genCrIdRsrcId2SiteMap(db);

  /* Binding functions for creating min-cost flow graph & maximum-flow graph */ {
    impl->genMinCostFlowGraphWrapper_ = [cnp, cr_id_rsrc_id2site_map, &db](auto &&group_array_wrapper) {
      return MinCostFlowGraphFactory::genMinCostFlowGraphFromPlaceDB(
              db,
              cr_id_rsrc_id2site_map,
              cnp,
              std::forward<decltype(group_array_wrapper)>(group_array_wrapper));
    };
    impl->genMaxFlowGraphWrapper_ = [cnp, cr_id_rsrc_id2site_map, &db](auto &&clock_info_wrapper) {
      return MaxFlowGraphFactory::genMaxFlowGraphFromPlaceDB(
              db,
              cnp,
              std::forward<decltype(clock_info_wrapper)>(clock_info_wrapper),
              cr_id_rsrc_id2site_map);
    };
  }

  impl->genClockRegionAssignmentResult_ = [cnp, &db](auto &&group_wrapper, auto &&group_result_wrapper) {
    return ClockRegionAssignmentResultFactory::genClockRegionAssignmentResult(
            std::forward<decltype(group_wrapper)>(group_wrapper),
            std::forward<decltype(group_result_wrapper)>(group_result_wrapper),
            cnp,
            db);
  };

  impl->commitResult_ = [cnp, &db](auto &&region_assignment_wrapper, auto &&cr_avail_crs) {
    return commitResult(cnp,
                        std::forward<decltype(region_assignment_wrapper)>(region_assignment_wrapper),
                        std::forward<decltype(cr_avail_crs)>(cr_avail_crs),
                        db);
  };

  impl->getClockNetToClockRegionBoxDist_ = [cnp, &db](auto &&ck_id, auto &&cr_box) {
    return getClockNetToClockRegionBoxDist(std::forward<decltype(ck_id)>(ck_id),
                                           std::forward<decltype(cr_box)>(cr_box),
                                           cnp,
                                           db);
  };

  return cnp;
}


void ClockNetworkPlannerFactory::initializeBins(
        PlaceDB &                                              db,
        std::vector<ClockNetworkPlannerImpl::IndexBox> &       bin_boxes_array,
        LayoutMap2D<ClockNetworkPlannerImpl::BoxIndexWrapper> &bin_id_size_map) {
  IndexType layout_width   = db.db()->layout().siteMap().width();
  IndexType layout_height  = db.db()->layout().siteMap().height();

  IndexType x_cnp_bin_size = db.cnpBinSizes().first;
  IndexType y_cnp_bin_size = db.cnpBinSizes().second;

  bin_boxes_array.clear();
  bin_id_size_map.reshape(layout_width, layout_height);
  for (int32_t i = 0; i < layout_width; i++) {
    for (int32_t j = 0; j < layout_height; j++) {
      bin_id_size_map.at(i, j).setId(std::numeric_limits<IndexType>::max());
    }
  }
  // Note that `ClockRegion::geometry_bbox` return the exclusive right boundary.
  for (const ClockRegion &cr_box : db.db()->layout().clockRegionMap()) {
    IndexType xl = cr_box.geometry_bbox().xl();
    IndexType xh = cr_box.geometry_bbox().xh();
    IndexType yl = cr_box.geometry_bbox().yl();
    IndexType yh = cr_box.geometry_bbox().yh();
    for (IndexType bin_xl = xl; bin_xl < xh; bin_xl += x_cnp_bin_size) {
      IndexType bin_xh = std::min(bin_xl + x_cnp_bin_size, xh);
      for (IndexType bin_yl = yl; bin_yl < yh; bin_yl += y_cnp_bin_size) {
        IndexType bin_yh = std::min(bin_yl + y_cnp_bin_size, yh);
        IndexType bin_id = bin_boxes_array.size();
        bin_boxes_array.emplace_back(bin_xl, bin_yl, bin_xh, bin_yh);
        auto &bin_box = bin_boxes_array.back();
        for (IndexType i = bin_xl; i < bin_xh; i++) {
          for (IndexType j = bin_yl; j < bin_yh; j++) {
            bin_id_size_map.at(i, j).setId(bin_id);
            bin_id_size_map.at(i, j).setRef(bin_box);
          }
        }
      }
    }
  }
  for (int32_t i = 0; i < layout_width; i++) {
    for (int32_t j = 0; j < layout_height; j++) {
      openparfAssert(bin_id_size_map.at(i, j).id() != std::numeric_limits<IndexType>::max());
    }
  }
}

std::tuple<RealType, RealType, IndexType, IndexType> ClockNetworkPlannerFactory::getClockNetToClockRegionBoxDist(
        IndexType                                   ck_id,
        const ClockNetworkPlannerFactory::IndexBox &cr_box,
        ClockNetworkPlanner                         cnp,
        PlaceDB &                                   db) {
  auto                          lo_cr_bbox = db.db()->layout().clockRegionMap().at(cr_box.xl(), cr_box.xh()).bbox();
  auto                          hi_cr_bbox = db.db()->layout().clockRegionMap().at(cr_box.xh(), cr_box.yh()).bbox();
  IndexBox                      site_box(lo_cr_bbox.xl(), lo_cr_bbox.yl(), hi_cr_bbox.xh(), hi_cr_bbox.yh());
  IndexType                     net_id                      = db.clockIdToNetId(ck_id);
  RealType                      dist_x                      = 0;
  RealType                      dist_y                      = 0;
  RealType                      non_zero_avg_euclidean_dist = 0;
  IndexType                     non_zero_count              = 0;
  std::unordered_set<IndexType> visited_insts;
  for (auto pin_id : db.netPins().at(net_id)) {
    openparfAssert(net_id == db.pin2Net(pin_id));
    IndexType inst_id = db.pin2Inst(pin_id);
    if (visited_insts.find(inst_id) != visited_insts.end()) {
      continue;
    }
    if (!db.isInstClockSource(inst_id)) {
      auto dist = site_box.manhDist(Point<RealType, 2>(cnp.inst_x_pos()[inst_id], cnp.inst_y_pos()[inst_id]));
      if (!(dist.x() == 0 && dist.y() == 0)) {
        non_zero_count++;
        non_zero_avg_euclidean_dist += std::sqrt(dist.x() * dist.x() + dist.y() * dist.y());
      }
      dist_x += 1.0 * db.instPins().at(inst_id).size() * dist.x();
      dist_y += 1.0 * db.instPins().at(inst_id).size() * dist.y();
      visited_insts.insert(inst_id);
    }
  }
  openparfAssert(!visited_insts.empty());
  double dist = dist_x * db.wirelengthWeights().first + dist_y * db.wirelengthWeights().second;
  if (non_zero_count > 0) {
    non_zero_avg_euclidean_dist /= non_zero_count;
  }
  return {dist, non_zero_avg_euclidean_dist, visited_insts.size(), non_zero_count};
}

}   // namespace clock_network_planner
