/**
 * @file   placedb.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 */


// C++ standard library headers
#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>


// project headers
#include "database/placedb.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

std::ostream &operator<<(std::ostream &os, ModelAreaType const &rhs) {
  os << className<decltype(rhs)>() << "(";
  os << "model_id_ : " << rhs.model_id_ << ", area_type_id_ : " << rhs.area_type_id_
     << ", model_size_ : " << rhs.model_size_;
  os << "\n)";
  return os;
}

std::ostream &operator<<(std::ostream &os, PlaceParams const &rhs) {
  os << className<decltype(rhs)>() << "(";
  os << "benchmark_name_ : " << rhs.benchmark_name_ << ", ";
  os << "area_type_names_ : " << rhs.area_type_names_ << ", model_area_types_ : " << rhs.model_area_types_
     << ", model_is_lut_ : " << rhs.model_is_lut_ << ", model_is_ff_ : " << rhs.model_is_ff_
     << ", model_is_clocksource_ : " << rhs.model_is_clocksource_
     << ", resource2area_types_ : " << rhs.resource2area_types_
     << ", resource_categories_ : " << rhs.resource_categories_ << ", CLB_capacity_ : " << rhs.CLB_capacity_
     << ", BLE_capacity_ : " << rhs.BLE_capacity_ << ", num_ControlSets_per_CLB_ : " << rhs.num_ControlSets_per_CLB_;
  os << "\n)";
  return os;
}

PlaceDB::PlaceDB(Database &db, PlaceParams place_params) : db_(db), place_params_(std::move(place_params)) {
  // must follow the order of initialization
  // initialize netlist information
  // after building the netlist, we use new ids for instances stored
  // in new2old_inst_ids_ and old2new_inst_ids_
  buildNetlist();

  if (place_params_.enable_carry_chain_packing_) {
    buildChain();
  }

  // initialize mapping between resources and area types
  buildResource2AreaTypeMapping();

  // initialize instance area types
  setInstAreaTypes();

  // initialize attributes
  setAttributes();

  // initialize layout information
  buildBinMaps();

  // instance areas
  setMovableFixedAreas();

  // Clock related dat structures.
  initClock();

  // Check fixed instances
  checkFixedInstToSiteRrscTypeCompatibility();
}

// std::vector<uint8_t> PlaceDB::resourceAreaTypes(std::string const &name) const {
//  return area_type_enum_.resourceAreaTypeIds(resource_type_enum_.alias(name));
//}

std::vector<uint8_t> PlaceDB::resourceAreaTypes(Resource const &resource) const {
  return place_params_.resource2area_types_[resource.id()];
}

std::vector<ModelAreaType> const &PlaceDB::modelAreaTypes(std::string const &name) const {
  return modelAreaTypes(db_->design().modelId(name));
}

std::vector<PlaceDB::IndexType> PlaceDB::collectInstIds(PlaceDB::IndexType resource_id) const {
  std::vector<IndexType> inst_ids;
  auto const            &design          = db_->design();
  auto const            &layout          = db_->layout();
  auto                   top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  for (auto const &old_inst_id : netlist.instIds()) {
    auto const &inst  = netlist.inst(old_inst_id);
    auto const &model = design.model(inst.attr().modelId());
    if (model.modelType() != ModelType::kModule) {
      // As one model may correspond to different resources.
      // Use the average area of different resources.
      auto const &resource_ids = layout.resourceMap().modelResourceIds(model.id());
      for (auto rid : resource_ids) {
        if (rid == resource_id) {
          inst_ids.push_back(newInstId(inst.id()));
          break;
        }
      }
    }
  }
  return inst_ids;
}

std::vector<PlaceDB::BoxType> PlaceDB::collectSiteBoxes(PlaceDB::IndexType resource_id) const {
  auto const          &design = db_->design();
  auto const          &layout = db_->layout();

  std::vector<BoxType> site_boxes;
  for (auto const &site : layout.siteMap()) {
    auto const &site_type = layout.siteType(site);
    if (site_type.resourceCapacity(resource_id)) {
      site_boxes.emplace_back(site.bbox().xl(), site.bbox().yl(), site.bbox().xh(), site.bbox().yh());
    }
  }

  return site_boxes;
}

std::vector<PlaceDB::BoxType> PlaceDB::collectSiteBoxes() const {
  auto const          &design = db_->design();
  auto const          &layout = db_->layout();

  std::vector<BoxType> site_boxes;
  site_boxes.reserve(layout.siteMap().size());
  for (auto const &site : layout.siteMap()) {
    site_boxes.emplace_back(site.bbox().xl(), site.bbox().yl(), site.bbox().xh(), site.bbox().yh());
  }

  return site_boxes;
}

std::vector<PlaceDB::CoordinateType> PlaceDB::collectFlattenSiteBoxes() const {
  std::vector<PlaceDB::BoxType> box_list = std::move(collectSiteBoxes());
  std::vector<CoordinateType>   flatten_list;
  for (auto const &box : box_list) {
    flatten_list.push_back(box.xl());
    flatten_list.push_back(box.yl());
    flatten_list.push_back(box.xh());
    flatten_list.push_back(box.yh());
  }
  return flatten_list;
}

std::vector<int32_t> PlaceDB::collectSiteCapacities(PlaceDB::IndexType resource_id) const {
  auto const          &design = db_->design();
  auto const          &layout = db_->layout();

  std::vector<int32_t> site_caps(layout.siteMap().size(), 0);
  for (auto const &site : layout.siteMap()) {
    auto const &site_type   = layout.siteType(site);
    site_caps.at(site.id()) = site_type.resourceCapacity(resource_id);
  }

  return site_caps;
}

int32_t PlaceDB::getSiteCapacity(IndexType site_id, IndexType resource_id) const {
  auto const &layout = db_->layout();
  auto const &site   = layout.siteMap().at(site_id);
  if (site.get() == nullptr) {
    // this site is not valid.
    return 0;
  }
  auto const &site_type = layout.siteType(site.value());
  return site_type.resourceCapacity(resource_id);
}

int32_t PlaceDB::getSiteCapacity(const Site &site, IndexType resource_id) const {
  auto const &layout    = db_->layout();
  auto const &site_type = layout.siteType(site);
  return site_type.resourceCapacity(resource_id);
}

PlaceDB::DimType PlaceDB::siteMapDim() const {
  auto const &design   = db_->design();
  auto const &layout   = db_->layout();

  auto const &site_map = layout.siteMap();
  return {site_map.width(), site_map.height()};
}

PlaceDB::BoxType PlaceDB::siteBBox(IndexType x, IndexType y) const {
  auto const &design   = db_->design();
  auto const &layout   = db_->layout();
  auto const &site_map = layout.siteMap();
  auto const  ptr      = site_map.at(x, y);
  BoxType     res;
  res.set(-1, -1, -1, -1);
  if (ptr) {
    res.set(ptr->bbox().xl(), ptr->bbox().yl(), ptr->bbox().xh(), ptr->bbox().yh());
  }
  return res;
}

PlaceDB::IndexType PlaceDB::XyToCrIndex(const PlaceDB::CoordinateType &x, const PlaceDB::CoordinateType &y) const {
  openparfAssert(0 <= x && x <= db_->layout().siteMap().width());
  openparfAssert(0 <= y && y <= db_->layout().siteMap().height());
  auto ix = std::min(static_cast<IndexType>(x), layout_to_clock_region_map.width() - 1);
  auto iy = std::min(static_cast<IndexType>(y), layout_to_clock_region_map.height() - 1);
  return layout_to_clock_region_map.at(ix, iy).ref().get().id();
}

PlaceDB::IndexType PlaceDB::XyToHcIndex(const PlaceDB::CoordinateType &x, const PlaceDB::CoordinateType &y) const {
  openparfAssert(0 <= x && x <= db_->layout().siteMap().width());
  openparfAssert(0 <= y && y <= db_->layout().siteMap().height());
  auto ix = std::min(static_cast<IndexType>(x), layout_to_half_column_map.width() - 1);
  auto iy = std::min(static_cast<IndexType>(y), layout_to_half_column_map.height() - 1);
  return layout_to_half_column_map.at(ix, iy).ref().get().id();
}

std::vector<int32_t> PlaceDB::XyToSlrIndex(const int32_t &x, const int32_t &y) const {
  std::vector<int32_t> XyIndex;
  int32_t              width  = static_cast<int32_t>(db_->layout().siteMap().width());
  int32_t              height = static_cast<int32_t>(db_->layout().siteMap().height());
  openparfAssert(0 <= x && x <= width);
  openparfAssert(0 <= y && y <= height);

  // We suppose all SLRs with a same width and height.
  IndexType num_slrX = numSlrX();
  IndexType num_slrY = numSlrY();
  openparfAssertMsg((width % num_slrX == 0) && (height % num_slrY == 0),
                    "width = %d, height = %d, num_slrX = %d, num_slrY = %d.\n",
                    width, height, num_slrX, num_slrY);

  int32_t slr_width  = width / num_slrX;
  int32_t slr_height = height / num_slrY;
  int32_t x_index    = x / slr_width;
  int32_t y_index    = y / slr_height;

  XyIndex.emplace_back(x_index);
  XyIndex.emplace_back(y_index);

  return XyIndex;
}

PlaceDB::IndexType PlaceDB::CrXyToSlrIndex(const int32_t &crX, const int32_t &crY) const {
  IndexType num_crX = numCrX();
  IndexType num_crY = numCrY();
  IndexType num_slrX = numSlrX();
  IndexType num_slrY = numSlrY();
  openparfAssert(0 <= crX && crX <= num_crX);
  openparfAssert(0 <= crY && crY <= num_crY);

  IndexType slrCrX = num_crX / num_slrX;
  IndexType slrCrY = num_crY / num_slrY;
  IndexType idx    = crX / slrCrX;
  IndexType idy    = crY / slrCrY;

  return idx * num_slrY + idy;
}

PlaceDB::IndexType PlaceDB::CrIndexToSlrIndex(const IndexType &crIdx) const {
  openparfAssertMsg(crIdx < numCr(), "The maxinum CR Id %d, but given CR Id %d\n", numCr(), crIdx);
  IndexType num_crX = numCrX();
  IndexType num_crY = numCrY();
  IndexType crX     = crIdx / num_crY;
  IndexType crY     = crIdx % num_crY;

  return CrXyToSlrIndex(crX, crY);
}

std::pair<PlaceDB::IndexType, PlaceDB::IndexType> PlaceDB::clockRegionMapSize() const {
  return std::make_pair(db_->layout().clockRegionMap().width(), db_->layout().clockRegionMap().height());
}

void PlaceDB::apply(std::vector<PlaceDB::Point3DType> const &locs) {
  auto &design          = db_->design();
  auto  top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto &netlist = top_module_inst->netlist();

  for (auto const &inst_id : netlist.instIds()) {
    auto &inst = netlist.inst(inst_id);
    if (inst.attr().placeStatus() == PlaceStatus::kMovable) {
      auto        new_inst_id = newInstId(inst_id);
      auto const &loc         = locs[new_inst_id];
      // update raw database
      inst.attr().setLoc(InstAttr::PointType(loc.x(), loc.y(), loc.z()));
      // update PlaceDB
      inst_locs_[new_inst_id] = loc;
    }
  }
}

bool PlaceDB::writeBookshelfPl(std::string const &pl_file) const {
  return db_->writeBookshelfPl(pl_file);
}

bool PlaceDB::writeBookshelfNodes(std::string const &nodes_file) const {
  return db_->writeBookshelfNodes(nodes_file);
}

bool PlaceDB::writeBookshelfNets(std::string const &nets_file) const {
  return db_->writeBookshelfNets(nets_file);
}

PlaceDB::IndexType PlaceDB::memory() const {
  IndexType ret =
          sizeof(db_) + sizeof(inst_pins_) + inst_pins_.memory() + sizeof(pin2inst_) +
          pin2inst_.capacity() * sizeof(decltype(pin2inst_)::value_type) + net_pins_.memory() + sizeof(pin2net_) +
          pin2net_.capacity() * sizeof(decltype(pin2net_)::value_type) +
          // sizeof(model_sizes_) +
          // model_sizes_.capacity() * sizeof(decltype(model_sizes_)::value_type) +
          sizeof(inst_sizes_) + inst_sizes_.capacity() * sizeof(decltype(inst_sizes_)::value_type) +
          sizeof(inst_area_types_) + inst_area_types_.capacity() * sizeof(decltype(inst_area_types_)::value_type) +
          sizeof(inst_place_status_) +
          inst_place_status_.capacity() * sizeof(decltype(inst_place_status_)::value_type) + sizeof(inst_locs_) +
          inst_locs_.capacity() * sizeof(decltype(inst_locs_)::value_type) + sizeof(pin_offsets_) +
          pin_offsets_.capacity() * sizeof(decltype(pin_offsets_)::value_type) + sizeof(net_weights_) +
          net_weights_.capacity() * sizeof(decltype(net_weights_)::value_type) + diearea_.memory() +
          sizeof(bin_map_dims_) + bin_map_dims_.capacity() * sizeof(decltype(bin_map_dims_)::value_type) +
          sizeof(bin_map_sizes_) + bin_map_sizes_.capacity() * sizeof(decltype(bin_map_sizes_)::value_type);
  for (auto const &bin_map : bin_capacity_maps_) {
    ret += bin_map.capacity() * sizeof(decltype(bin_capacity_maps_)::value_type::value_type);
  }
  // a very rough estimation
  // ret += sizeof(area_type_enum_);

  ret += sizeof(total_movable_inst_areas_) +
         total_movable_inst_areas_.capacity() * sizeof(decltype(total_movable_inst_areas_)::value_type) +
         sizeof(total_fixed_inst_areas_) +
         total_fixed_inst_areas_.capacity() * sizeof(decltype(total_fixed_inst_areas_)::value_type) +
         sizeof(new2old_inst_ids_) + new2old_inst_ids_.capacity() * sizeof(decltype(new2old_inst_ids_)::value_type) +
         sizeof(old2new_inst_ids_) + old2new_inst_ids_.capacity() * sizeof(decltype(old2new_inst_ids_)::value_type) +
         sizeof(movable_range_) + sizeof(fixed_range_) + sizeof(resource_num_sites_) +
         resource_num_sites_.capacity() * sizeof(decltype(resource_num_sites_)::value_type) +
         sizeof(resource_avg_site_sizes_) +
         resource_avg_site_sizes_.capacity() * sizeof(decltype(resource_avg_site_sizes_)::value_type) +
         sizeof(resource_max_site_sizes_) +
         resource_max_site_sizes_.capacity() * sizeof(decltype(resource_max_site_sizes_)::value_type);
  return ret;
}

// void PlaceDB::copy(PlaceDB const &rhs) {
//  db_ = rhs.db_;
//
//  inst_pins_ = rhs.inst_pins_;
//  pin2inst_ = rhs.pin2inst_;
//  net_pins_ = rhs.net_pins_;
//  pin2net_ = rhs.pin2net_;
//
//  model_sizes_ = rhs.model_sizes_;
//  inst_sizes_ = rhs.inst_sizes_;
//  inst_place_status_ = rhs.inst_place_status_;
//  inst_locs_ = rhs.inst_locs_;
//  pin_offsets_ = rhs.pin_offsets_;
//  pin_signal_types_ = rhs.pin_signal_types_;
//  pin_signal_directs_ = rhs.pin_signal_directs_;
//  net_weights_ = rhs.net_weights_;
//
//  diearea_ = rhs.diearea_;
//  bin_map_dims_ = rhs.bin_map_dims_;
//  bin_map_sizes_ = rhs.bin_map_sizes_;
//  bin_capacity_maps_ = rhs.bin_capacity_maps_;
//
//  resource_type_enum_ = rhs.resource_type_enum_;
//  inst_resource_types_ = rhs.inst_resource_types_;
//  area_type_enum_ = rhs.area_type_enum_;
//  inst_area_types_ = rhs.inst_area_types_;
//  area_type_inst_groups = rhs.area_type_inst_groups;
//
//  total_movable_inst_areas_ = rhs.total_movable_inst_areas_;
//  total_fixed_inst_areas_ = rhs.total_fixed_inst_areas_;
//
//  new2old_inst_ids_ = rhs.new2old_inst_ids_;
//  old2new_inst_ids_ = rhs.old2new_inst_ids_;
//  movable_range_ = rhs.movable_range_;
//  fixed_range_ = rhs.fixed_range_;
//
//  resource_num_sites_ = rhs.resource_num_sites_;
//  resource_avg_site_sizes_ = rhs.resource_avg_site_sizes_;
//  resource_max_site_sizes_ = rhs.resource_max_site_sizes_;
//}
//
// void PlaceDB::move(PlaceDB &&rhs) {
//  db_ = std::move(rhs.db_);
//
//  inst_pins_ = std::move(rhs.inst_pins_);
//  pin2inst_ = std::move(rhs.pin2inst_);
//  net_pins_ = std::move(rhs.net_pins_);
//  pin2net_ = std::move(rhs.pin2net_);
//
//  model_sizes_ = std::move(rhs.model_sizes_);
//  inst_sizes_ = std::move(rhs.inst_sizes_);
//  inst_place_status_ = std::move(rhs.inst_place_status_);
//  inst_locs_ = std::move(rhs.inst_locs_);
//  pin_offsets_ = std::move(rhs.pin_offsets_);
//  pin_signal_types_ = std::move(rhs.pin_signal_types_);
//  pin_signal_directs_ = std::move(rhs.pin_signal_directs_);
//  net_weights_ = std::move(rhs.net_weights_);
//
//  diearea_ = std::move(rhs.diearea_);
//  bin_map_dims_ = std::move(rhs.bin_map_dims_);
//  bin_map_sizes_ = std::move(rhs.bin_map_sizes_);
//  bin_capacity_maps_ = std::move(rhs.bin_capacity_maps_);
//
//  resource_type_enum_ = std::move(rhs.resource_type_enum_);
//  inst_resource_types_ = std::move(rhs.inst_resource_types_);
//  area_type_enum_ = std::move(rhs.area_type_enum_);
//  inst_area_types_ = std::move(rhs.inst_area_types_);
//  area_type_inst_groups = std::move(rhs.area_type_inst_groups);
//
//  total_movable_inst_areas_ = std::move(rhs.total_movable_inst_areas_);
//  total_fixed_inst_areas_ = std::move(rhs.total_fixed_inst_areas_);
//
//  new2old_inst_ids_ = std::move(rhs.new2old_inst_ids_);
//  old2new_inst_ids_ = std::move(rhs.old2new_inst_ids_);
//  movable_range_ = std::move(rhs.movable_range_);
//  fixed_range_ = std::move(rhs.fixed_range_);
//
//  resource_num_sites_ = std::move(rhs.resource_num_sites_);
//  resource_avg_site_sizes_ = std::move(rhs.resource_avg_site_sizes_);
//  resource_max_site_sizes_ = std::move(rhs.resource_max_site_sizes_);
//}

std::ostream &operator<<(std::ostream &os, PlaceDB const &rhs) {
  os << className<decltype(rhs)>() << "(";
  os << "\ndb_ : " << rhs.db_->id() << ", \ninst_pins_ : " << rhs.inst_pins_ << ", \npin2inst_ : " << rhs.pin2inst_
     << ", \nnet_pins_ : " << rhs.net_pins_ << ", \npin2net_ : "
     << rhs.pin2net_
     //<< ", \nmodel_sizes_ : " << rhs.model_sizes_
     << ", \ninst_sizes_ : " << rhs.inst_sizes_ << ", \ninst_area_types_ : " << rhs.inst_area_types_
     << ", \ninst_place_status_ : " << rhs.inst_place_status_ << ", \ninst_locs_ : " << rhs.inst_locs_
     << ", \npin_offsets_ : " << rhs.pin_offsets_ << ", \nnet_weights_ : " << rhs.net_weights_
     << ", \ndiearea_ : " << rhs.diearea_ << ", \nbin_map_dims_ : " << rhs.bin_map_dims_
     << ", \nbin_map_sizes_ : " << rhs.bin_map_sizes_ << ", \nbin_capacity_maps_ : " << rhs.bin_capacity_maps_;
  // os << ", resource_type_enum_ : " << rhs.resource_type_enum_;
  // os << ", area_type_enum_ : " << rhs.area_type_enum_;
  os << ", \ntotal_movable_inst_areas_ : " << rhs.total_movable_inst_areas_;
  os << ", \ntotal_fixed_inst_areas_ : " << rhs.total_fixed_inst_areas_;
  os << ", \nnew2old_inst_ids_ : " << rhs.new2old_inst_ids_;
  os << ", \nold2new_inst_ids_ : " << rhs.old2new_inst_ids_;
  os << ", \nmovable_range_ : "
     << "(" << rhs.movable_range_.first << ", " << rhs.movable_range_.second << ")";
  os << ", \nfixed_range_ : "
     << "(" << rhs.fixed_range_.first << ", " << rhs.fixed_range_.second << ")";
  os << ", \nresource_num_sites_ : " << rhs.resource_num_sites_;
  os << ", \nresource_avg_site_sizes_ : " << rhs.resource_avg_site_sizes_;
  os << ", \nresource_max_site_sizes_ : " << rhs.resource_max_site_sizes_;
  os << "\n)";
  return os;
}

void PlaceDB::sortInstsByPlaceStatus() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const            &netlist = top_module_inst->netlist();

  std::vector<IndexType> order2inst_ids(netlist.numInsts());
  std::iota(order2inst_ids.begin(), order2inst_ids.end(), 0);

  std::sort(order2inst_ids.begin(), order2inst_ids.end(), [&](IndexType a, IndexType b) {
    auto const &inst_a = netlist.inst(a);
    auto const &inst_b = netlist.inst(b);
    auto        key_a  = toInteger(inst_a.attr().placeStatus());
    auto        key_b  = toInteger(inst_b.attr().placeStatus());
    return key_a < key_b || (key_a == key_b && inst_a.id() < inst_b.id());
  });

  std::vector<IndexType> inst_id2orders(netlist.numInsts());
  for (IndexType i = 0; i < netlist.numInsts(); ++i) {
    inst_id2orders[order2inst_ids[i]] = i;
  }

  // oldInstId() and newInstId() are ready to use after this
  new2old_inst_ids_.swap(order2inst_ids);
  old2new_inst_ids_.swap(inst_id2orders);

  fixed_range_.first  = netlist.numInsts();
  fixed_range_.second = netlist.numInsts();
  for (IndexType new_inst_id = 0; new_inst_id < netlist.numInsts(); ++new_inst_id) {
    auto        old_inst_id = oldInstId(new_inst_id);
    auto const &inst        = netlist.inst(old_inst_id);
    if (inst.attr().placeStatus() == PlaceStatus::kFixed) {
      fixed_range_.first = new_inst_id;
      break;
    }
  }
  movable_range_.first  = 0;
  movable_range_.second = fixed_range_.first;

  // check sorting
  for (IndexType new_inst_id = movable_range_.first; new_inst_id < movable_range_.second; ++new_inst_id) {
    auto        old_inst_id = oldInstId(new_inst_id);
    auto const &inst        = netlist.inst(old_inst_id);
    openparfAssert(inst.attr().placeStatus() == PlaceStatus::kMovable);
  }
  for (IndexType new_inst_id = fixed_range_.first; new_inst_id < fixed_range_.second; ++new_inst_id) {
    auto        old_inst_id = oldInstId(new_inst_id);
    auto const &inst        = netlist.inst(old_inst_id);
    openparfAssert(inst.attr().placeStatus() == PlaceStatus::kFixed);
  }
}

void PlaceDB::buildNetlist() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  // sort instances
  sortInstsByPlaceStatus();

  // initialize netlist information

  // build inst_pins_ and pin2inst_
  inst_pins_.reserve(netlist.numInsts(), netlist.numPins());
  pin2inst_.resize(netlist.numPins());
  IndexType count = 0;
  // must traverse in new order
  for (auto inst_id : new2old_inst_ids_) {
    auto const &inst = netlist.inst(inst_id);
    inst_pins_.pushBackIndexBegin(count);
    for (auto const &pin_id : inst.pinIds()) {
      inst_pins_.pushBack(pin_id);
      // must convert to new ids
      pin2inst_.at(pin_id) = newInstId(inst.id());
      ++count;
    }
  }

  // build net_pins_ and pin2net_
  net_pins_.reserve(netlist.numNets(), netlist.numPins());
  pin2net_.resize(netlist.numPins());
  count = 0;
  for (auto const &net_id : netlist.netIds()) {
    auto const &net = netlist.net(net_id);
    net_pins_.pushBackIndexBegin(count);
    for (auto const &pin_id : net.pinIds()) {
      net_pins_.pushBack(pin_id);
      pin2net_.at(pin_id) = net.id();
      ++count;
    }
  }

  // check if have one-pin net
  for (auto const &net_id : netlist.netIds()) {
    auto const &net = netlist.net(net_id);
    if (net.pinIds().size() == 1) {
      openparfPrint(kWarn, "Net %s have only one pins.\n", net.attr().name().c_str());
    }
  }

  // initialize inst_id and inst name mapping
  IndexType newId = 0;
  for (auto inst_id : new2old_inst_ids_) {
    auto const &inst = netlist.inst(inst_id);
    auto       &n    = inst.attr().name();
    openparfAssert(name_to_inst.find(n) == name_to_inst.end());
    name_to_inst.insert({n, newId});
    ++newId;
  }

  for (auto const &net_id : netlist.netIds()) {
    auto const &net = netlist.net(net_id);
    openparfAssert(net.id() == net_id);
    name_to_net.insert({net.attr().name(), net.id()});
  }
}

namespace {

std::vector<PlaceDB::IndexType> ExtractChainFromOneInst(PlaceDB &placedb,
        std::vector<bool>                                       &visited_mark,
        Design::IndexType                                        module_id,
        const database::Inst                                    &source_inst) {
  using database::Inst;
  using database::Model;
  using database::Net;
  using database::Pin;
  const Design                   &design          = placedb.db()->design();
  const Layout                   &layout          = placedb.db()->layout();
  auto                            top_module_inst = design.topModuleInst();
  auto const                     &netlist         = top_module_inst->netlist();
  const Model                    &model           = design.model(module_id);
  std::vector<PlaceDB::IndexType> ordinal_inst_ids;
  std::vector<PlaceDB::IndexType> reversed_inst_ids;
  std::queue<PlaceDB::IndexType>  que;
  auto IsVisited       = [&visited_mark](const Inst &inst) -> bool { return visited_mark[inst.id()] == true; };
  auto IsChain         = [&module_id](const Inst &inst) -> bool { return inst.attr().modelId() == module_id; };
  auto SetVisited      = [&visited_mark](const Inst &inst) { visited_mark[inst.id()] = true; };
  auto IsCascadedInPin = [&model](const Pin &pin) {
    const auto &model_pin = model.modelPin(pin.modelPinId());
    return model_pin.signalDirect() == SignalDirection::kInput && model_pin.signalType() == SignalType::kCascade;
  };
  auto IsCascadedOutPin = [&model](const Pin &pin) {
    const auto &model_pin = model.modelPin(pin.modelPinId());
    return model_pin.signalDirect() == SignalDirection::kOutput && model_pin.signalType() == SignalType::kCascade;
  };

  SetVisited(source_inst);
  openparfAssert(IsChain(source_inst));

  /* only search fomr cascaded output pin to cascaded input pin */ {
    ordinal_inst_ids.clear();
    que.push(source_inst.id());
    ordinal_inst_ids.push_back(source_inst.id());
    while (!que.empty()) {
      auto        inst_id = que.front();
      const Inst &inst    = netlist.inst(inst_id);
      que.pop();
      for (auto pin_id : inst.pinIds()) {
        const Pin &pin = netlist.pin(pin_id);
        if (!IsCascadedOutPin(pin)) continue;
        const Net &net = netlist.net(pin.netId());
        for (auto adjacent_pin_id : net.pinIds()) {
          const Pin  &adjacent_pin     = netlist.pin(adjacent_pin_id);
          auto        adjacent_inst_id = adjacent_pin.instId();
          const Inst &adjacent_inst    = netlist.inst(adjacent_inst_id);
          if (!IsChain(adjacent_inst) || IsVisited(adjacent_inst) || !IsCascadedInPin(adjacent_pin)) {
            continue;
          }
          SetVisited(adjacent_inst);
          que.push(adjacent_inst_id);
          ordinal_inst_ids.push_back(adjacent_inst_id);
        }
      }
    }
  }

  /* only search form cascaded input pin to cascaded output pin */ {
    reversed_inst_ids.clear();
    que.push(source_inst.id());
    while (!que.empty()) {
      auto        inst_id = que.front();
      const Inst &inst    = netlist.inst(inst_id);
      que.pop();
      for (auto pin_id : inst.pinIds()) {
        const Pin &pin = netlist.pin(pin_id);
        if (!IsCascadedInPin(pin)) continue;
        const Net &net = netlist.net(pin.netId());
        for (auto adjacent_pin_id : net.pinIds()) {
          const Pin  &adjacent_pin     = netlist.pin(adjacent_pin_id);
          auto        adjacent_inst_id = adjacent_pin.instId();
          const Inst &adjacent_inst    = netlist.inst(adjacent_inst_id);
          if (!IsChain(adjacent_inst) || IsVisited(adjacent_inst) || !IsCascadedOutPin(adjacent_pin)) {
            continue;
          }
          SetVisited(adjacent_inst);
          que.push(adjacent_inst_id);
          reversed_inst_ids.push_back(adjacent_inst_id);
        }
      }
    }
  }

  // combine the instances in two directions.
  std::reverse(reversed_inst_ids.begin(), reversed_inst_ids.end());
  ordinal_inst_ids.insert(ordinal_inst_ids.end(), reversed_inst_ids.begin(), reversed_inst_ids.end());


  for (int32_t i = 0; i < ordinal_inst_ids.size(); i++) {
    int32_t     inst_id = ordinal_inst_ids[i];
    const Inst &inst    = netlist.inst(inst_id);
  }
  return ordinal_inst_ids;
}

}   // namespace

void PlaceDB::buildChain() {
  const Design      &design          = db_->design();
  const Layout      &layout          = db_->layout();
  auto               top_module_inst = design.topModuleInst();
  auto const        &netlist         = top_module_inst->netlist();
  const std::string &module_name     = place_params_.carry_chain_module_name_;
  Design::IndexType  module_id       = design.modelId(module_name);
  openparfAssert(module_id != std::numeric_limits<decltype(module_id)>::max());
  IndexType         num_inst = netlist.numInsts();
  IndexType         count    = 0;
  std::vector<bool> visited_mark(num_inst, false);
  auto              IsVisited = [&visited_mark](const Inst &inst) -> bool { return visited_mark[inst.id()] == true; };
  auto              IsChain   = [&module_id](const Inst &inst) -> bool { return inst.attr().modelId() == module_id; };

  chain_insts_.clear();

  openparfPrint(kDebug, "carry_chain_module_name: %s(%i)\n", module_name.c_str(), module_id);

  for (IndexType old_inst_id : new2old_inst_ids_) {
    const Inst            &inst = netlist.inst(old_inst_id);
    std::vector<IndexType> chain_old_inst_ids;
    if (!IsChain(inst) || IsVisited(inst)) continue;
    chain_old_inst_ids = ExtractChainFromOneInst(*this, visited_mark, module_id, inst);
    chain_insts_.pushBackIndexBegin(count);
    for (auto const &old_inst_id : chain_old_inst_ids) {
      auto new_inst_id = old2new_inst_ids_[old_inst_id];
      chain_insts_.pushBack(new_inst_id);
    }
    count += chain_old_inst_ids.size();
  }

  // dump the extracted carry chain.
  if (false) {
    std::ofstream of("carry_chain.txt");
    for (int i = 0; i < chain_insts_.size1(); i++) {
      for (int j = 0; j < chain_insts_.size2(i); j++) {
        auto        new_inst_id = chain_insts_.at(i, j);
        auto        old_inst_id = new2old_inst_ids_[new_inst_id];
        const Inst &inst        = netlist.inst(old_inst_id);
        of << inst.attr().name() << "(" << new_inst_id << ") ";
      }
      of << std::endl;
    }
    of.close();
  }
}

void PlaceDB::buildResource2AreaTypeMapping() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  // compute the average and maximum site sizes of each resource type
  resource_num_sites_.assign(layout.resourceMap().numResources(), 0);
  resource_avg_site_sizes_.assign(layout.resourceMap().numResources(), {0, 0});
  resource_max_site_sizes_.assign(layout.resourceMap().numResources(), {1, 1});
  for (auto const &site : layout.siteMap()) {
    auto const &site_type = layout.siteType(site);
    for (IndexType resource_id = 0; resource_id < site_type.numResources(); ++resource_id) {
      auto resource_capacity = site_type.resourceCapacity(resource_id);
      if (resource_capacity) {
        auto &max_site = resource_max_site_sizes_[resource_id];
        max_site.setX(std::max((CoordinateType) max_site.x(), (CoordinateType) site.bbox().width()));
        max_site.setY(std::max((CoordinateType) max_site.y(), (CoordinateType) site.bbox().height()));
        auto &avg_site = resource_avg_site_sizes_[resource_id];
        avg_site.setX(avg_site.x() + site.bbox().width());
        avg_site.setY(avg_site.y() + site.bbox().height());
        resource_num_sites_[resource_id] += 1;
      }
    }
  }
  for (IndexType resource_id = 0; resource_id < layout.resourceMap().numResources(); ++resource_id) {
    resource_avg_site_sizes_[resource_id].set(
            resource_avg_site_sizes_[resource_id].x() / resource_num_sites_[resource_id],
            resource_avg_site_sizes_[resource_id].y() / resource_num_sites_[resource_id]);
  }

  ////////////////////////////////////////////////////////////////////////
  // This is rather a hard-coded way to map resource type to area type
  // Here I list typical settings in FPGA
  // SITEs: SLICEL, SLICEM, DSP, RAM, IO
  // RESOURCEs: LUTL, LUTM, FF, CARRY, DSP, RAM, IO
  // The problem comes from LUTL and LUTM.
  // SLICEL often contains {LUTL, FF, CARRY};
  // SLICEM often contains {LUTM, FF, CARRY}.
  // LUTL can host LUT1-6, while
  // LUTM can host {LUT1-6, LRAM, SHIFT}, which makes things complicated.
  // Considering that SLICEL and SLICEM usually appear in an interleaved way,
  // e.g., 68 SLICELs and 60 SLICEMs, I regard them as one kind of SLICE.
  // This means that there are only one area type called LUT.
  // The areas for LRAM and SHIFT are increased as LUTL cannot host them.
  ////////////////////////////////////////////////////////////////////////

  //// build resource to area type maping
  // for (auto const &resource : layout.resourceMap()) {
  //  resource_type_enum_.tryExtend(resource.name());
  //}
  // area_type_enum_.initialize(resource_type_enum_);

  // model_sizes_.assign(design.numModels(), {0, 0});
  // for (auto const &model : design.models()) {
  //  if (model.modelType() != ModelType::kModule) {
  //    openparfAssert(model_sizes_map.count(model.name()));
  //    auto const& sizes = model_sizes_map.at(model.name());
  //    openparfAssert(sizes.size() == 2U);
  //    model_sizes_[model.id()].set(sizes[0], sizes[1]);
  //  }
  //}

  //// print resource to area mapping
  // openparfPrint(kInfo, "-------------------------------------\n");
  // openparfPrint(kInfo, "Resource <=> Area type Mapping\n");
  // for (auto const &resource : layout.resourceMap()) {
  //  auto const &ats = resourceAreaTypes(resource.name());
  //  openparfPrint(kInfo, "%s (%u) =>", resource.name().c_str(),
  //                resource_type_enum_.id(resource.name()));
  //  for (auto const &at : ats) {
  //    openparfPrint(kNone, " %s (%u)", area_type_enum_.name(at).c_str(),
  //                  (IndexType)at);
  //  }
  //  openparfPrint(kNone, "\n");
  //}

  //// print model sizes
  // openparfPrint(kInfo, "-------------------------------------\n");
  // openparfPrint(kInfo, "Model sizes\n");
  // for (auto const &model : design.models()) {
  //  if (model.modelType() != ModelType::kModule) {
  //    openparfPrint(kInfo, "%s (%u) => (%g, %g)\n", model.name().c_str(),
  //                  model.id(), model_sizes_[model.id()].width(),
  //                  model_sizes_[model.id()].height());
  //  }
  //}
}

void PlaceDB::setInstAreaTypes() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  inst_resource_types_.resize(netlist.numInsts());
  inst_area_types_.resize(netlist.numInsts());
  is_inst_luts_.resize(netlist.numInsts());
  is_inst_ffs_.resize(netlist.numInsts());
  is_inst_dsps_.resize(netlist.numInsts());
  is_inst_rams_.resize(netlist.numInsts());
  is_inst_clocksource_.resize(netlist.numInsts());

  area_type_inst_groups.resize(numAreaTypes());
  for (auto const &inst_id : netlist.instIds()) {
    auto const &inst                  = netlist.inst(inst_id);
    auto const &model                 = design.model(inst.attr().modelId());
    auto        new_inst_id           = newInstId(inst.id());
    is_inst_luts_[new_inst_id]        = place_params_.model_is_lut_[model.id()];
    is_inst_ffs_[new_inst_id]         = place_params_.model_is_ff_[model.id()];
    is_inst_clocksource_[new_inst_id] = place_params_.model_is_clocksource_[model.id()];

    // set area type
    // we handle LUT/LUTL/LUTM in a special way
    // make sure there are no other situations not considered
    auto const &resource_ids          = layout.resourceMap().modelResourceIds(model.id());
    // collect instance resources
    for (auto rid : resource_ids) {
      inst_resource_types_[new_inst_id].push_back(rid);
    }
    // collect instances of this area type
    for (auto const &at : place_params_.model_area_types_[model.id()]) {
      inst_area_types_[new_inst_id].push_back(at.area_type_id_);
      area_type_inst_groups[at.area_type_id_].push_back(new_inst_id);
    }
    //// a model in both resources LUTL and LUTM
    // if (resource_ids.size() == 2U &&
    //    ((resource_type_enum_.alias(
    //          layout.resourceMap().resource(resource_ids[0]).name()) ==
    //          "LUTL" &&
    //      resource_type_enum_.alias(
    //          layout.resourceMap().resource(resource_ids[1]).name()) ==
    //          "LUTM") ||
    //     (resource_type_enum_.alias(
    //          layout.resourceMap().resource(resource_ids[1]).name()) ==
    //          "LUTL" &&
    //      resource_type_enum_.alias(
    //          layout.resourceMap().resource(resource_ids[0]).name()) ==
    //          "LUTM"))) {
    //  inst_area_types_[new_inst_id] = resourceAreaTypes("LUTL");
    //} else {
    //  for (auto resource_id : resource_ids) {
    //    auto const &resource = layout.resourceMap().resource(resource_id);
    //    auto area_types = resourceAreaTypes(resource.name());
    //    if (inst_area_types_[new_inst_id].empty()) {
    //      inst_area_types_[new_inst_id] = area_types;
    //    } else {
    //      openparfAssertMsg(
    //          inst_area_types_[new_inst_id] == area_types,
    //          "inconsistent area types for instance %s, resource %s",
    //          inst.attr().name().c_str(), resource.name().c_str());
    //    }
    //  }
    //}
  }
  // LUT must have output pin
  for (auto const &new_inst_id : netlist.instIds()) {
    if (!is_inst_luts_[new_inst_id]) {
      continue;
    }
    auto        inst_id              = new2old_inst_ids_[new_inst_id];
    auto const &inst                 = netlist.inst(inst_id);
    bool        have_output_pin_flag = false;
    for (auto const &pin_id : inst.pinIds()) {
      auto const &pin = netlist.pin(pin_id);
      if (pin.attr().signalDirect() == SignalDirection::kOutput) {
        have_output_pin_flag = true;
        break;
      }
    }
    if (!have_output_pin_flag) {
      openparfPrint(kWarn, "LUT %s does not have output pin.\n", inst.attr().name().c_str());
    }
  }
}

void PlaceDB::setAttributes() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  inst_sizes_.resize(netlist.numInsts() * numAreaTypes(), {0, 0});
  inst_place_status_.resize(netlist.numInsts());
  inst_locs_.resize(netlist.numInsts());
  for (auto const &inst_id : netlist.instIds()) {
    auto const &inst        = netlist.inst(inst_id);
    auto const &model       = design.model(inst.attr().modelId());
    auto        new_inst_id = newInstId(inst.id());
    for (auto const &at : modelAreaTypes(model.id())) {
      inst_sizes_[new_inst_id * numAreaTypes() + at.area_type_id_] = at.model_size_;
    }
    inst_place_status_[new_inst_id] = inst.attr().placeStatus();
    inst_locs_[new_inst_id].set(inst.attr().loc().x(), inst.attr().loc().y(), inst.attr().loc().z());
  }

  pin_offsets_.resize(netlist.numPins());
  pin_signal_directs_.resize(netlist.numPins());
  pin_signal_types_.resize(netlist.numPins());
  for (auto const &pin_id : netlist.pinIds()) {
    auto const &pin         = netlist.pin(pin_id);
    auto const &inst        = netlist.inst(pin.instId());
    auto const &model       = design.model(inst.attr().modelId());
    auto const &model_pin   = model.modelPin(pin.modelPinId());
    auto        new_inst_id = newInstId(inst.id());
    SizeType    max_size    = {0, 0};
    for (auto const &at : modelAreaTypes(model.id())) {
      max_size.set(std::max(max_size.x(), at.model_size_.x()), std::max(max_size.y(), at.model_size_.y()));
    }
    pin_offsets_[pin.id()].set(0.5 * max_size.x(), 0.5 * max_size.y());
    pin_signal_directs_[pin.id()] = model_pin.signalDirect();
    pin_signal_types_[pin.id()]   = model_pin.signalType();
  }

  net_weights_.resize(netlist.numNets());
  for (auto const &net_id : netlist.netIds()) {
    auto const &net        = netlist.net(net_id);
    net_weights_[net.id()] = net.attr().weight();
  }
}

void PlaceDB::buildBinMaps() {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();

  // determine the dimensions of maps
  diearea_.set(0, 0, layout.siteMap().width(), layout.siteMap().height());
  bin_map_dims_.resize(numAreaTypes());
  bin_map_sizes_.resize(bin_map_dims_.size());
  std::vector<uint8_t> area_type_markers(bin_map_sizes_.size(), 0);
  // compute the nearest number in power of 2
  auto                 nearest_power2   = [](CoordinateType v) { return std::pow(2, std::round(std::log2(v))); };
  auto                 ceil_power2      = [](CoordinateType v) { return std::pow(2, std::ceil(std::log2(v))); };
  IndexType            basic_num_bins_x = ceil_power2(diearea_.width());
  IndexType            basic_num_bins_y = ceil_power2(diearea_.height());
  IndexType            count            = 0;
  for (auto const &resource : layout.resourceMap()) {
    // auto area_types = resourceAreaTypes(resource.name());
    auto area_types = resourceAreaTypes(resource);
    for (auto area_type : area_types) {
      // only set one area type once
      if (!area_type_markers[area_type]) {
        auto &dims = bin_map_dims_[area_type];
        dims.set(ceil_power2(diearea_.width() / resource_max_site_sizes_[resource.id()].x()),
                nearest_power2(diearea_.height() / resource_max_site_sizes_[resource.id()].y()));
        bin_map_sizes_[area_type] = {diearea_.width() / dims.x(), diearea_.height() / dims.y()};
        count += dims.product();
      }
    }
  }
  // initialize the maps
  bin_capacity_maps_.resize(bin_map_dims_.size());
  for (IndexType i = 0; i < bin_map_dims_.size(); ++i) {
    auto       &bin_map = bin_capacity_maps_[i];
    auto const &dims    = bin_map_dims_[i];
    bin_map.assign(dims.product(), 0);
  }
  for (auto const &site : layout.siteMap()) {
    auto const &site_type = layout.siteType(site);
    BoxType     site_box(site.bbox().xl(), site.bbox().yl(), site.bbox().xh(), site.bbox().yh());
    for (IndexType resource_id = 0; resource_id < site_type.numResources(); ++resource_id) {
      auto resource_capacity = site_type.resourceCapacity(resource_id);
      if (resource_capacity) {
        // auto area_types = resourceAreaTypes(
        //    layout.resourceMap().resource(resource_id).name());
        auto area_types = resourceAreaTypes(layout.resourceMap().resource(resource_id));
        for (auto area_type : area_types) {
          auto const &dims    = bin_map_dims_[area_type];
          auto const &sizes   = bin_map_sizes_[area_type];
          auto       &bin_map = bin_capacity_maps_.at(area_type);
          IndexType   bxl     = std::min((IndexType) ((site.bbox().xl() - diearea_.xl()) * dims.x() / diearea_.width()),
                        (IndexType) (dims.x() - 1));
          IndexType   byl = std::min((IndexType) ((site.bbox().yl() - diearea_.yl()) * dims.y() / diearea_.height()),
                    (IndexType) (dims.y() - 1));
          IndexType   bxh =
                  std::min((IndexType) std::ceil((site.bbox().xh() - diearea_.xl()) * dims.x() / diearea_.width()),
                          (IndexType) (dims.x()));
          IndexType byh =
                  std::min((IndexType) std::ceil((site.bbox().yh() - diearea_.yl()) * dims.y() / diearea_.height()),
                          (IndexType) (dims.y()));

          foreach2D(bxl, byl, bxh, byh, [&](IndexType ix, IndexType iy) {
            BoxType bin_box = {diearea_.xl() + ix * sizes.width(), diearea_.yl() + iy * sizes.height(),
                    diearea_.xl() + (ix + 1) * sizes.width(), diearea_.yl() + (iy + 1) * sizes.height()};
            bin_map.at(ix * dims.y() + iy) += bin_box.overlap(site_box);
          });
        }
      }
    }
  }
}

void PlaceDB::setMovableFixedAreas() {
  total_movable_inst_areas_.assign(numAreaTypes(), 0);
  total_fixed_inst_areas_.assign(numAreaTypes(), 0);
  for (IndexType i = 0; i < numInsts(); ++i) {
    auto const &area_types = instAreaType(i);
    for (auto area_type : area_types) {
      switch (instPlaceStatus(i)) {
        case PlaceStatus::kMovable:
          total_movable_inst_areas_[area_type] += instSize(i, area_type).product();
          break;
        case PlaceStatus::kFixed:
          total_fixed_inst_areas_[area_type] += instSize(i, area_type).product();
          break;
        default:
          openparfAssert(0);
      }
    }
  }
}

void PlaceDB::initClock() {
  // First, we generate net to clock index mapping and net to clock source id mapping
  // clock index is the id for clocks, ranging from 0 to num_clock_nets - 1
  clock_net_indexes_.resize(numNets());
  net_to_clocksource_.resize(numNets());
  std::fill(clock_net_indexes_.begin(), clock_net_indexes_.end(), IndexType(InvalidIndex<IndexType>::value));
  std::fill(net_to_clocksource_.begin(), net_to_clocksource_.end(), IndexType(InvalidIndex<IndexType>::value));
  for (IndexType i = 0; i < numInsts(); ++i) {
    // This is a clock source instance
    // We consider the net of its output pin a clock net.
    // Note: this is NOT the same as net with a kClock pin
    if (isInstClockSource(i)) {
      auto const &pins            = inst_pins_.at(i);
      auto const  num_pins        = pins.size();
      bool        have_output_pin = false;
      for (IndexType p = 0; p < num_pins; p++) {
        IndexType pin_id     = inst_pins_.at(i, p);
        auto      signal_dir = pinSignalDirect(pin_id);
        // TODO(Jing Mai): should kINPUTOUTPUT count?
        if (signal_dir == SignalDirection::kOutput) {
          auto net_id = pin2net_[pin_id];
          openparfAssertMsg(
                  net_to_clocksource_[net_id] == InvalidIndex<IndexType>::value || net_to_clocksource_[net_id] == i,
                  "A clock net can only have one clock source instance connected to it.");
          net_to_clocksource_[net_id] = i;
          have_output_pin             = true;
          clock_net_indexes_[net_id]  = 1;
        }
      }
      openparfAssertMsg(have_output_pin,
              "A clock source instance must have at least one output pin. Instance %i does not", i);
    }
  }
  IndexType num_clock_nets = 0;
  for (IndexType i = 0; i < numNets(); i++) {
    if (clock_net_indexes_[i] == 1) {
      // find a new clock net
      clock_net_indexes_[i] = num_clock_nets++;
      clock_id_to_clock_net_id_.push_back(i);
    }
  }
  num_clock_nets_ = num_clock_nets;
  openparfAssert(clock_id_to_clock_net_id_.size() == num_clock_nets_);

  // Now we generate instance to clock index mapping
  inst_clock_indexes_.resize(numInsts());
  for (IndexType i = 0; i < numInsts(); i++) {
    auto const         &pins     = inst_pins_.at(i);
    IndexType           num_pins = pins.size();
    // Each instance can be connected to more than one clock net
    std::set<IndexType> s;
    for (IndexType j = 0; j < num_pins; j++) {
      IndexType clock_idx = clock_net_indexes_[pin2net_[inst_pins_.at(i, j)]];
      if (clock_idx != InvalidIndex<IndexType>::value) {
        s.insert(clock_idx);
      }
    }
    if (s.empty()) {
      inst_clock_indexes_[i].clear();
    } else {
      inst_clock_indexes_[i].assign(s.begin(), s.end());
    }
  }

  IndexType width  = db_->layout().siteMap().width();
  IndexType height = db_->layout().siteMap().height();

  layout_to_clock_region_map.reshape(width, height);
  layout_to_half_column_map.reshape(width, height);

  // Generate layout-to-clock-region mapping
  if (place_params_.honor_clock_region_constraints) {
    uint32_t cr_id = 0;
    for (const ClockRegion &clock_region : db_->layout().clockRegionMap()) {
      auto xl = static_cast<IndexType>(clock_region.bbox().xl());
      auto yl = static_cast<IndexType>(clock_region.bbox().yl());
      auto xh = static_cast<IndexType>(clock_region.bbox().xh());
      auto yh = static_cast<IndexType>(clock_region.bbox().yh());
      for (IndexType ix = xl; ix <= xh; ix++) {
        for (IndexType iy = yl; iy <= yh; iy++) {
          openparfAssertMsg(&layout_to_clock_region_map.at(ix, iy).ref().get() == &ClockRegionRefType::NullValue,
                  "Clock regions are overlapped");
          layout_to_clock_region_map.at(ix, iy).setRef(clock_region);
        }
      }
    }
    // Ensure that any location on the layout is mapped.
    for (ClockRegionRefType &site : layout_to_clock_region_map) {
      openparfAssert(&site.ref().get() != &ClockRegionRefType::NullValue);
    }
  }
  if (place_params_.honor_half_column_constraints) {
    // Generate layout-to-half-column mapping
    std::vector<HalfColumnRegion> const &half_columns = db_->layout().halfColumnRegions();
    for (const HalfColumnRegion &half_column : half_columns) {
      auto               xl           = static_cast<IndexType>(half_column.bbox().xl());
      auto               yl           = static_cast<IndexType>(half_column.bbox().yl());
      auto               xh           = static_cast<IndexType>(half_column.bbox().xh());
      auto               yh           = static_cast<IndexType>(half_column.bbox().yh());
      IndexType          par_cr_id    = half_column.clockRegionId();
      const ClockRegion &clock_region = db_->layout().clockRegionMap().at(par_cr_id);
      IndexType          hc_xmin      = clock_region.HcXMin();
      IndexType          hc_xmax      = clock_region.HcXMAx();
      auto               cr_xl        = static_cast<IndexType>(clock_region.bbox().xl());
      auto               cr_xh        = static_cast<IndexType>(clock_region.bbox().xh());
      /**
       * within a clock region, There may be gaps on the left and right boundary that do not belong
       * to any half column. Here we assign them to the nearest half columns.
       */
      if (xl == hc_xmin) {
        xl = cr_xl;
      }
      if (xh == hc_xmax) {
        xh = cr_xh;
      }
      for (IndexType ix = xl; ix <= xh; ix++) {
        for (IndexType iy = yl; iy <= yh; iy++) {
          openparfAssertMsg(&layout_to_half_column_map.at(ix, iy).ref().get() == &HalfColumnRegionRefType::NullValue,
                  "Half columns are overlapped");
          layout_to_half_column_map.at(ix, iy).setRef(half_column);
        }
      }
    }
    for (IndexType ix = 0; ix < layout_to_half_column_map.width(); ix++) {
      for (IndexType iy = 0; iy < layout_to_half_column_map.height(); iy++) {
        HalfColumnRegionRefType site = layout_to_half_column_map.at(ix, iy);
        openparfAssertMsg(&site.ref().get() != &HalfColumnRegionRefType::NullValue,
                "Site (%d, %d) does not belong to any half column\n", ix, iy);
      }
    }
  }
}

void PlaceDB::checkFixedInstToSiteRrscTypeCompatibility() const {
  auto const &design          = db_->design();
  auto const &layout          = db_->layout();
  auto        top_module_inst = design.topModuleInst();
  openparfAssert(top_module_inst);
  // assume flat netlist for now
  auto const &netlist = top_module_inst->netlist();
  // check the site fixed instance
  for (IndexType new_inst_id = fixed_range_.first; new_inst_id < fixed_range_.second; ++new_inst_id) {
    auto        old_inst_id = oldInstId(new_inst_id);
    auto const &inst        = netlist.inst(old_inst_id);
    IndexType   inst_x      = inst.attr().loc().x();
    IndexType   inst_y      = inst.attr().loc().y();
    auto site_proxy = layout.siteMap().at(inst_x, inst_y);
    openparfAssertMsg(site_proxy, "Site(%d, %d) not exists\n", inst_x, inst_y);
    openparfAssertMsg(instToSiteRrscTypeCompatibility(new_inst_id, *site_proxy),
                      "Fixed inst %d is not compatible in Site(%d, %d)", new_inst_id, inst_x, inst_y);
  }
}

bool PlaceDB::instToSiteRrscTypeCompatibility(IndexType inst_id, IndexType site_id) const {
  auto &inst_rsrc_ids = instResourceType(inst_id);
  for (const auto &rsrc_id : inst_rsrc_ids) {
    if (getSiteCapacity(site_id, rsrc_id) == 0) {
      return false;
    }
  }
  return true;
}

bool PlaceDB::instToSiteRrscTypeCompatibility(IndexType inst_id, const Site &site) const {
  auto &inst_rsrc_ids = instResourceType(inst_id);
  for (const auto &rsrc_id : inst_rsrc_ids) {
    if (getSiteCapacity(site, rsrc_id) == 0) {
      return false;
    }
  }
  return true;
}

}   // namespace database


OPENPARF_END_NAMESPACE
