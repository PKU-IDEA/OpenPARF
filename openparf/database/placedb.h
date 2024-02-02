/**
 * File              : placedb.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 08.26.2021
 * Last Modified Date: 08.26.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */

#ifndef OPENPARF_DATABASE_PLACEDB_H_
#define OPENPARF_DATABASE_PLACEDB_H_

#include <unordered_map>

#include "container/container.hpp"
#include "container/index_wrapper.hpp"
#include "database/clock_region.h"
#include "database/database.h"
#include "database/half_column_region.h"
#include "database/layout_map.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {


struct ModelAreaType {
  using CoordinateType = float;
  using IndexType      = CoordinateTraits<CoordinateType>::IndexType;
  using SizeType       = geometry::Size<CoordinateType, 2>;

  IndexType            model_id_;       ///< model index
  IndexType            area_type_id_;   ///< area type index
  SizeType             model_size_;     ///< model size on this area type

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, ModelAreaType const &rhs);
};

struct PlaceParams {
  using CoordinateType = float;
  using IndexType      = CoordinateTraits<CoordinateType>::IndexType;

  // benchmark name
  std::string                                benchmark_name_;   ///< benchmark name

  // area types
  std::vector<std::string>                   area_type_names_;   ///< names of area types
  std::unordered_map<std::string, IndexType> at_name2id_map_;    ///< map string name to area type index

  // models
  std::vector<std::vector<ModelAreaType>>    model_area_types_;       ///< map model to area types
  std::vector<uint8_t>                       model_is_lut_;           ///< whether a model is a LUT, if yes, the number
                                                                      ///< indicates the number of pins
  std::vector<uint8_t>                       model_is_ff_;            ///< whether a model is a FF
  std::vector<uint8_t>                       model_is_clocksource_;   ///< whether a model is a ClockSource. Might not
                                                                      ///< have meaning depending on architecture
  // resources
  std::vector<std::vector<uint8_t>>          resource2area_types_;   ///< map resources to area types
  std::vector<ResourceCategory>              resource_categories_;   ///< resource categories of a resource

  // area adjustment
  std::vector<int32_t>                       adjust_area_at_ids_;   ///< area type ids that should be considered in area
                                                                    ///< adjustment phase

  // architecture
  ModelAreaType::IndexType                   CLB_capacity_;
  ModelAreaType::IndexType                   BLE_capacity_;
  ModelAreaType::IndexType                   num_ControlSets_per_CLB_;

  CoordinateType                             x_wirelength_weight_;
  CoordinateType                             y_wirelength_weight_;

  IndexType                                  x_cnp_bin_size_;
  IndexType                                  y_cnp_bin_size_;

  // clock related
  bool                                       honor_clock_region_constraints;
  bool                                       honor_half_column_constraints;

  IndexType                                  clock_region_capacity_;
  IndexType                                  cnp_utplace2_enable_packing_;
  IndexType                                  cnp_utplace2_max_num_clock_pruning_per_iteration_;


  // carry chain related
  bool                                       enable_carry_chain_packing_;   ///< whether enable carry chain packing
  std::string                                carry_chain_module_name_;      ///< carry chain model name
  std::string                                ssr_chain_module_name_;        ///< ssr chain model name

  /// @brief overload output stream
  friend std::ostream                       &operator<<(std::ostream &os, PlaceParams const &rhs);
};

/// @brief Database for placement.
class PlaceDB {
 public:
  using CoordinateType          = ModelAreaType::CoordinateType;
  using IndexType               = CoordinateTraits<CoordinateType>::IndexType;
  using WeightType              = CoordinateTraits<CoordinateType>::WeightType;
  using SizeType                = geometry::Size<CoordinateType, 2>;
  using Point2DType             = geometry::Point<CoordinateType, 2>;
  using Point3DType             = geometry::Point<CoordinateType, 3>;
  using DimType                 = geometry::Size<IndexType, 2>;
  using BoxType                 = geometry::Box<CoordinateType>;
  using ClockRegionRefType      = container::IndexWrapper<const ClockRegion, IndexType>;
  using HalfColumnRegionRefType = container::IndexWrapper<const HalfColumnRegion, IndexType>;

  /// @brief constructor
  PlaceDB(Database &db, PlaceParams place_params);

  ///// @brief copy constructor
  // PlaceDB(PlaceDB const &rhs) { copy(rhs); }

  ///// @brief move constructor
  // PlaceDB(PlaceDB &&rhs) noexcept { move(std::move(rhs)); }

  ///// @brief copy assignment
  // PlaceDB &operator=(PlaceDB const &rhs) {
  //  if (this != &rhs) {
  //    copy(rhs);
  //  }
  //  return *this;
  //}

  ///// @brief move assignment
  // PlaceDB &operator=(PlaceDB &&rhs) noexcept {
  //  if (this != &rhs) {
  //    move(std::move(rhs));
  //  }
  //  return *this;
  //}

  /// @brief getter for raw database
  container::ObserverPtr<const Database> db() const { return container::ObserverPtr<const Database>(db_.get()); }

  PlaceParams const                     &place_params() const { return place_params_; }

  /// @brief number of instances
  IndexType                              numInsts() const { return inst_pins_.size1(); }

  /// @brief number of nets
  IndexType                              numNets() const { return net_pins_.size1(); }

  /// @brief number of pins
  IndexType                              numPins() const { return pin2inst_.size(); }

  /// @brief getter for inst_pins_
  container::FlatNestedVector<IndexType, IndexType> const &instPins() const { return inst_pins_; }

  /// @brief getter for pin2inst_
  std::vector<IndexType> const                            &pin2Inst() const { return pin2inst_; }

  /// @brief getter for instance given one pin
  IndexType                                                pin2Inst(IndexType id) const {
                                                   openparfAssert(id < numPins());
                                                   return pin2inst_[id];
  }

  /// @brief getter for net_pins_
  container::FlatNestedVector<IndexType, IndexType> const &netPins() const { return net_pins_; }

  /// @brief getter for pin2net_
  std::vector<IndexType> const                            &pin2Net() const { return pin2net_; }

  /// @brief getter for net given one pin
  IndexType                                                pin2Net(IndexType id) const {
                                                   openparfAssert(id < numPins());
                                                   return pin2net_[id];
  }

  /// @brief getter for inst_sizes_
  std::vector<SizeType> const &instSizes() const { return inst_sizes_; }

  /// @brief getter for instance size
  SizeType                     instSize(IndexType id, IndexType at) const {
                        openparfAssert(id < numInsts());
                        openparfAssert(at < numAreaTypes());
                        return inst_sizes_[id * numAreaTypes() + at];
  }


  /// @brief getter for inst_resource_types_
  std::vector<std::vector<uint8_t>> const &instResourceTypes() const { return inst_resource_types_; }

  /// @brief getter for instance resource type
  std::vector<uint8_t> const              &instResourceType(IndexType id) const {
                 openparfAssert(id < numInsts());
                 return inst_resource_types_[id];
  }

  /// @brief Whether the resource type set of a instance is the sub-set of that of a site.
  bool      instToSiteRrscTypeCompatibility(IndexType inst_id, IndexType site_id) const;

  /// @brief Whether the resource type set of a instance is the sub-set of that of a site.
  bool      instToSiteRrscTypeCompatibility(IndexType inst_id, const Site &site) const;

  /**
   * @brief Get the model index of any instance.
   * @param id instance index
   * @return model index
   */
  IndexType getInstModelId(IndexType id) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &inst    = netlist.inst(oldInstId(id));
    return inst.attr().modelId();
  }

  /**
   * @brief Get the array indicating the model indexs of all instances.
   */
  std::vector<IndexType> getInstModelIds() const {
    std::vector<IndexType> flags(numInsts());
    for (IndexType i = 0; i < numInsts(); ++i) {
      flags[i] = getInstModelId(i);
    }
    return flags;
  }

  /**
   * @brief Get area type index from area type name
   * @param name A string indicating the area type name
   * @return area type index.
   */
  IndexType getAreaTypeIndexFromName(const std::string &at_name) const {
    auto rv = place_params_.at_name2id_map_.find(at_name);
    openparfAssert(rv != place_params_.at_name2id_map_.end());
    return rv->second;
  }

  /// @brief check whether an instance is a LUT
  uint8_t                     isInstLUT(IndexType id) const { return is_inst_luts_[id]; }

  /// @brief array of flags indicating whether an instance is a LUT
  const std::vector<uint8_t> &isInstLUTs() const { return is_inst_luts_; }

  /// @brief check whether an instance is FF
  bool                        isInstFF(IndexType id) const { return (is_inst_ffs_[id] != 0); }

  bool                        isInstDSP(IndexType id) const { return (is_inst_dsps_[id] != 0); }

  bool                        isInstRAM(IndexType id) const { return (is_inst_rams_[id] != 0); }

  /// @brief array of flags indicating whether an instance is a FF
  const std::vector<uint8_t> &isInstFFs() const { return is_inst_ffs_; }

  bool                        isInstClockSource(IndexType id) const { return (is_inst_clocksource_[id] != 0); }


  /// @brief check whether a resource contains LUT
  bool                        isResourceLUT(IndexType id) const {
                           openparfAssert(id < place_params_.resource2area_types_.size());
                           auto const &area_types = place_params_.resource2area_types_[id];
                           for (auto const &area_type : area_types) {
                             if (place_params_.area_type_names_[area_type] == "LUT") {
                               return true;
      }
    }
                           return false;
  }

  /// @brief check whether a resource contains FF
  bool isResourceFF(IndexType id) const {
    openparfAssert(id < place_params_.resource2area_types_.size());
    auto const &area_types = place_params_.resource2area_types_[id];
    for (auto const &area_type : area_types) {
      if (place_params_.area_type_names_[area_type] == "FF") {
        return true;
      }
    }
    return false;
  }

  IndexType numClockNets() const { return num_clock_nets_; }

  IndexType numHalfColumnRegions() const {
    auto const &layout = db_->layout();
    return layout.numHalfColumnRegions();
  }


  /// @brief Return a net to clock index mapping
  /// @return a #(num_nets) vector res. If for net i, the value of res[i]
  /// isn't InvalidIndex<IndexType>::value, then a) i is a clock net, and
  /// b) the clock index of net i is res[i]. The clock index is a value between
  /// 0 and numClockNets - 1.
  const std::vector<IndexType> &getClockNetIndex() const { return clock_net_indexes_; }

  /// @brief Given the clock id of a clock net, return its net id
  IndexType                     clockIdToNetId(IndexType clockId) const {
                        openparfAssert(clockId >= 0 and clockId < num_clock_nets_);
                        return clock_id_to_clock_net_id_[clockId];
  }

  /// @brief Given the net id of a clock net, return its clock id
  /// @return The clock id of the clock net with net id netId
  ///         If the given net is not a clock net, then InvalidIndex<IndexType>::value
  ///         is returned
  IndexType netIdToClockId(IndexType netId) const {
    openparfAssert(netId >= 0 and netId < numNets());
    return clock_net_indexes_[netId];
  }

  /// @brief Return a net to clock source mapping
  /// @return a #(num_nets) vector res. If for net i, the value of res[i]
  /// isn't InvalidIndex<IndexType>::value, then a) i is a clock net, and
  /// b) the instance id of the clock source instance connected to net i is res[i].
  /// Note: this depends on the assumption that each clock net connects to only one
  /// clock source instance
  const std::vector<IndexType>              &getNetClockSourceMapping() const { return net_to_clocksource_; }

  /// @brief Return a mapping between instance id and the clock indexes it connects to
  /// @return a #(num_insts) vector res. res[i] contains the clock indexes of the clock nets connected to instance i.
  /// Each res[i] are guaranteed to be sorted with smallest clock index first. Empty means the instance is not a clock
  /// instance
  const std::vector<std::vector<IndexType>> &instToClocks() const { return inst_clock_indexes_; }
  std::vector<std::vector<IndexType>>        instToClocksCP() const { return inst_clock_indexes_; }
  /// @brief Retrieve the name of an instance from its (new) inst id
  /// @return Name of the instance, as found in bookshelf file
  /// This can be used to faciliate loading placement data from other placer
  std::string                                instName(IndexType id) const {
                                   auto const &design          = db_->design();
                                   auto        top_module_inst = design.topModuleInst();
                                   openparfAssert(top_module_inst);
                                   // assume flat netlist for now
                                   auto const &netlist = top_module_inst->netlist();
                                   auto const &inst    = netlist.inst(oldInstId(id));
                                   return inst.attr().name();
  }

  // @brief Retrieve the model name of an instance from its (new) inst id
  // @return Name of the model, as found in bookshelf file
  std::string instModelName(IndexType id) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &inst    = netlist.inst(oldInstId(id));
    auto const &model   = design.model(inst.attr().modelId());
    return model.name();
  }

  IndexType   nameToInst(std::string name) const { return name_to_inst.at(name); }

  std::string netName(IndexType id) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &net     = netlist.net(id);
    return net.attr().name();
  }
  IndexType                     nameToNet(std::string name) const { return name_to_net.at(name); }

  /// @brief get the resource category of an instance
  std::vector<ResourceCategory> instResourceCategory(IndexType id) const {
    auto const &design          = db_->design();
    auto const &layout          = db_->layout();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const                   &netlist      = top_module_inst->netlist();
    auto const                   &inst         = netlist.inst(oldInstId(id));
    auto const                   &model        = design.model(inst.attr().modelId());
    auto const                   &resource_ids = layout.resourceMap().modelResourceIds(model.id());
    std::vector<ResourceCategory> category;
    for (auto resource_id : resource_ids) {
      category.push_back(place_params_.resource_categories_[resource_id]);
    }
    return category;
  }

  /// @brief array of resource categories for instances
  std::vector<std::vector<ResourceCategory>> instResourceCategories() const {
    std::vector<std::vector<ResourceCategory>> flags(numInsts());
    for (IndexType i = 0; i < numInsts(); ++i) {
      flags[i] = instResourceCategory(i);
    }
    return flags;
  }

  /// @brief resource category for a resource
  ResourceCategory              resourceCategory(IndexType id) const { return place_params_.resource_categories_[id]; }

  /// @brief array of resource categories for resources
  std::vector<ResourceCategory> resourceCategories() const { return place_params_.resource_categories_; }

  // @brief area type ids that should be considered in area adjustment phase
  std::vector<int32_t>         &adjust_area_at_ids() { return place_params_.adjust_area_at_ids_; }

  /// @brief number of LUT/FF pairs in a CLB
  IndexType                     CLBCapacity() const { return place_params_.CLB_capacity_; }

  /// @brief number of LUT/FF pairs in a BLE
  IndexType                     BLECapacity() const { return place_params_.BLE_capacity_; }

  std::pair<PlaceParams::CoordinateType, PlaceParams::CoordinateType> wirelengthWeights() const {
    return {place_params_.x_wirelength_weight_, place_params_.y_wirelength_weight_};
  }

  std::pair<IndexType, IndexType> cnpBinSizes() const {
    return {place_params_.x_cnp_bin_size_, place_params_.y_cnp_bin_size_};
  }

  IndexType ClockRegionCapacity() const { return place_params_.clock_region_capacity_; };

  IndexType cnpUtplace2EnablePacking() const { return place_params_.cnp_utplace2_enable_packing_; }

  IndexType cnpUtplace2MaxNumClockPruningPerIteration() const {
    return place_params_.cnp_utplace2_max_num_clock_pruning_per_iteration_;
  }


  /// @brief number of Clock/ControlSetSR/ControlSetCE tuples in a CLB
  IndexType numControlSetsPerCLB() const { return place_params_.num_ControlSets_per_CLB_; }

  /// @brief number of LUT/FF pairs in a half CLB
  IndexType halfCLBCapacity() const { return CLBCapacity() / 2; }

  /// @brief number of BLEs in a CLB
  IndexType numBLEsPerCLB() const { return CLBCapacity() / BLECapacity(); }

  /// @brief number of BLEs in a half CLB
  IndexType numBLEsPerHalfCLB() const { return numBLEsPerCLB() / 2; }

  ///// @brief getter for resource type enum
  // ResourceTypeEnum const& resourceTypeEnum() const {
  //  return resource_type_enum_;
  //}

  /// @brief getter for inst_area_types_
  std::vector<std::vector<uint8_t>> const &instAreaTypes() const { return inst_area_types_; }

  /// @brief getter for instance area type
  std::vector<uint8_t> const              &instAreaType(IndexType id) const {
                 openparfAssertMsg(id < numInsts(), "%u < %u", id, numInsts());
                 return inst_area_types_[id];
  }

  ///@brief getter for instance area_type_names, converts area type id to string name
  std::string const areaTypeName(uint8_t id) const { return place_params_.area_type_names_[id]; }

  /// @brief getter for instances grouped by area types
  std::vector<std::vector<IndexType>> const &areaTypeInstGroups() const { return area_type_inst_groups; }

  /// @brief getter for instances of one area type
  std::vector<IndexType> const              &areaTypeInstGroup(IndexType area_type) const {
                 openparfAssert(area_type < numAreaTypes());
                 return area_type_inst_groups[area_type];
  }

  /// @brief getter for inst_place_status_
  std::vector<PlaceStatus> const &instPlaceStatus() const { return inst_place_status_; }

  /// @brief getter for instance placement status
  PlaceStatus const              &instPlaceStatus(IndexType id) const {
                 openparfAssert(id < numInsts());
                 return inst_place_status_[id];
  }

  /// @brief getter for inst_locs_
  std::vector<Point3DType> const     &instLocs() const { return inst_locs_; }

  /// @brief get offset of one pin
  Point2DType                         pinOffset(IndexType id) const { return pin_offsets_[id]; }

  /// @brief getter for pin_offsets_
  std::vector<Point2DType> const     &pinOffsets() const { return pin_offsets_; }

  /// @brief get signal direction of one pin
  SignalDirection                     pinSignalDirect(IndexType id) const { return pin_signal_directs_[id]; }

  /// @brief getter for pin_signal_directs_
  std::vector<SignalDirection> const &pinSignalDirects() const { return pin_signal_directs_; }

  /// @brief get signal type of one pin
  SignalType                          pinSignalType(IndexType id) const { return pin_signal_types_[id]; }

  /// @brief getter for pin_signal_types_
  std::vector<SignalType> const      &pinSignalTypes() const { return pin_signal_types_; }

  /// @brief (cell, model pin name) to pin id in inst
  IndexType                           cellPinMoelNameToPinId(IndexType inst_id, std::string cell_pin_name) const {
                              auto const &design          = db_->design();
                              auto        top_module_inst = design.topModuleInst();
                              openparfAssert(top_module_inst);
                              // assume flat netlist for now
                              auto const &netlist       = top_module_inst->netlist();
                              auto const &inst          = netlist.inst(oldInstId(inst_id));
                              auto const &attr          = inst.attr();
                              auto const &model         = design.model(attr.modelId());
                              auto        mpin_id       = model.modelPinId(cell_pin_name);
                              auto const &pin_ids       = inst.pinIds();
                              int32_t     target_pin_id = -1;
                              for (auto pin_id : pin_ids) {
                                auto const &pin = netlist.pin(pin_id);
                                if (pin.modelPinId() == mpin_id) {
                                  target_pin_id = pin_id;
                                  break;
      }
    }
                              openparfAssert(target_pin_id != -1);
                              return target_pin_id;
  }

  /// @brief pin name to pin model id in inst
  IndexType pinNameToPinModelId(IndexType inst_id, std::string cell_pin_name) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &inst    = netlist.inst(oldInstId(inst_id));
    auto const &attr    = inst.attr();
    auto const &model   = design.model(attr.modelId());
    auto        mpin_id = model.modelPinId(cell_pin_name);
    return mpin_id;
  }

  IndexType pinidToPinModelId(IndexType pin_id) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &pin     = netlist.pin(pin_id);
    return pin.modelPinId();
  }

  std::string pinidToPinModelName(IndexType pin_id) const {
    auto const &design          = db_->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist   = top_module_inst->netlist();
    auto const &pin       = netlist.pin(pin_id);
    auto const &inst      = netlist.inst(pin.instId());
    auto const &model     = design.model(inst.attr().modelId());
    auto const &model_pin = model.modelPin(pin.modelPinId());
    return model_pin.name();
  }


  /// @brief getter for net_weights_
  WeightType                     netWeight(IndexType id) const { return net_weights_[id]; }

  /// @brief getter for net_weights_
  std::vector<WeightType> const &netWeights() const { return net_weights_; }

  /// @brief getter for diearea_
  BoxType const                 &diearea() const { return diearea_; }

  /// @brief getter for bin_map_dims_
  std::vector<DimType> const    &binMapDims() const { return bin_map_dims_; }

  /// @brief getter for dimension of specific area type
  DimType const                 &binMapDim(IndexType area_type) const {
                    openparfAssert(area_type < bin_map_dims_.size());
                    return bin_map_dims_[area_type];
  }

  /// @brief getter for bin_map_sizes_
  std::vector<SizeType> const &binMapSizes() const { return bin_map_sizes_; }

  /// @brief getter for size of specific area type
  SizeType const              &binMapSize(IndexType area_type) const {
                 openparfAssert(area_type < bin_map_sizes_.size());
                 return bin_map_sizes_[area_type];
  }

  /// @brief getter for bin capacity maps
  std::vector<std::vector<CoordinateType>> const &binCapacityMaps() const { return bin_capacity_maps_; }

  /// @brief getter for the bin capacity map of one area type
  std::vector<CoordinateType> const              &binCapacityMap(IndexType area_type) const {
                 openparfAssert(area_type < bin_capacity_maps_.size());
                 return bin_capacity_maps_[area_type];
  }

  /// @brief number of resources
  IndexType numResources() const { return place_params_.resource2area_types_.size(); }

  /// @brief number of area types
  IndexType numAreaTypes() const {
    // return area_type_enum_.size();
    return place_params_.area_type_names_.size();
  }

  /// @brief get area type from a resource name
  // std::vector<uint8_t> resourceAreaTypes(std::string const &name) const;

  /// @brief get area type from a resource
  std::vector<uint8_t>              resourceAreaTypes(Resource const &resource) const;

  /// @brief get the sizes for a model given index
  std::vector<ModelAreaType> const &modelAreaTypes(IndexType id) const {
    // openparfAssert(id < model_sizes_.size());
    // return model_sizes_[id];
    openparfAssert(id < place_params_.model_area_types_.size());
    return place_params_.model_area_types_[id];
  }

  /// @brief get the sizes for a model given name
  std::vector<ModelAreaType> const      &modelAreaTypes(std::string const &name) const;

  /// @brief compute total movable instance areas
  std::vector<CoordinateType> const     &totalMovableInstAreas() const { return total_movable_inst_areas_; }

  /// @brief compute total fixed instance areas
  std::vector<CoordinateType> const     &totalFixedInstAreas() const { return total_fixed_inst_areas_; }

  /// @brief get index range for movable instances
  std::pair<IndexType, IndexType> const &movableRange() const { return movable_range_; }

  /// @brief get index range for fixed instances
  std::pair<IndexType, IndexType> const &fixedRange() const { return fixed_range_; }

  /// @brief get the old instance id given new instance id
  IndexType                              oldInstId(IndexType new_id) const {
                                 openparfAssertMsg(new_id < new2old_inst_ids_.size(), "%u < %lu", new_id, new2old_inst_ids_.size());
                                 return new2old_inst_ids_[new_id];
  }

  /// @brief get the new instance id given old instance id
  IndexType newInstId(IndexType old_id) const {
    openparfAssertMsg(old_id < old2new_inst_ids_.size(), "%u < %lu", old_id, old2new_inst_ids_.size());
    return old2new_inst_ids_[old_id];
  }

  /// @brief collect instances given a resource
  std::vector<IndexType>      collectInstIds(IndexType resource_id) const;


  // =============================================
  // [BEGIN  ] >>>>>>>>>>  Site attribute >>>>>>>>>>
  // =============================================

  /// @brief collect sites given a resource
  std::vector<BoxType>        collectSiteBoxes(IndexType resource_id) const;

  /// @brief all site boxes
  std::vector<BoxType>        collectSiteBoxes() const;

  /// @brief collect sites boxes and return as a flatten list
  std::vector<CoordinateType> collectFlattenSiteBoxes() const;

  /// @brief site capacities given a resource
  std::vector<int32_t>        collectSiteCapacities(IndexType resource_id) const;

  /// @brief get resource capacity for a site
  int32_t                     getSiteCapacity(IndexType site_id, IndexType resource_id) const;

  /// @brief get resource capacity for a site
  int32_t                     getSiteCapacity(const Site &site, IndexType resource_id) const;

  /// @brief site map dimensions
  DimType                     siteMapDim() const;

  BoxType                     siteBBox(IndexType x, IndexType y) const;

  IndexType                   XyToCrIndex(const CoordinateType &x, const CoordinateType &y) const;

  IndexType                   XyToHcIndex(const CoordinateType &x, const CoordinateType &y) const;

  std::vector<int32_t>        XyToSlrIndex(const int32_t &x, const int32_t &y) const;

  IndexType                   CrXyToSlrIndex(const int32_t &crX, const int32_t &crY) const;

  IndexType                   CrIndexToSlrIndex(const IndexType &crIdx) const;

  IndexType numCrX() const {
    auto const &layout = db_->layout();
    return layout.clockRegionMap().width();
  }

  IndexType numCrY() const {
    auto const &layout = db_->layout();
    return layout.clockRegionMap().height();
  }

  IndexType numCr() const { return numCrX() * numCrY(); }

  std::pair<IndexType, IndexType>                          clockRegionMapSize() const;

  // =============================================
  // [  END] <<<<<<<<<<  Site attribute <<<<<<<<<<
  // =============================================

  // =============================================
  // [BEGIN ] >>>>>>> CarryChain Attribute >>>>>>>
  // =============================================

  const container::FlatNestedVector<IndexType, IndexType> &chainInsts() const { return chain_insts_; }

  // =============================================
  // [  END] <<<<<<<< CarryChain Attribute <<<<<<<
  // =============================================

  // =============================================
  // [BEGIN ] >>>>>>> Multi-Die Attribute >>>>>>>
  // =============================================
  IndexType numSlrX() const {
    auto const &layout = db_->layout();
    return layout.superLogicRegionMap().width();
  }

  IndexType numSlrY() const {
    auto const &layout = db_->layout();
    return layout.superLogicRegionMap().height();
  }

  IndexType numSlr() const { return numSlrX() * numSlrY(); }

  IndexType slrWidth() const { return db_->layout().superLogicRegionMap().at(0).width(); }

  IndexType slrHeight() const { return db_->layout().superLogicRegionMap().at(0).height(); }

  // =============================================
  // [  END] <<<<<<<< Multi-Die Attribute <<<<<<<
  // =============================================

  /// @brief apply placement solution
  /// @param locs (x, y, z) locations given in new instance ids
  void                                                     apply(std::vector<Point3DType> const &locs);

  /// @brief write bookshelf .pl file
  bool                                                     writeBookshelfPl(std::string const &pl_file) const;

  /// @brief write bookshelf .nodes file
  bool                                                     writeBookshelfNodes(std::string const &nodes_file) const;

  /// @brief write bookshelf .nets file
  bool                                                     writeBookshelfNets(std::string const &nets_file) const;

  /// @brief summarize memory usage of the object in bytes
  IndexType                                                memory() const;

 protected:
  ///// @brief copy object
  // void copy(PlaceDB const &rhs);
  ///// @brief move object
  // void move(PlaceDB &&rhs);
  /// @brief overload output stream
  friend std::ostream                              &operator<<(std::ostream &os, PlaceDB const &rhs);

  /// @brief sort instances from movable to fixed
  /// mapping between old and new instance indices are built
  void                                              sortInstsByPlaceStatus();
  /// @brief build netlist and form flat data structure
  void                                              buildNetlist();
  /// @brief build carry chain and form flat data structure
  void                                              buildChain();
  /// @brief derive mapping between resource and area types
  void                                              buildResource2AreaTypeMapping();
  /// @brief set instance areas
  void                                              setInstAreaTypes();
  /// @brief set attributes for instances, nets, and pins
  void                                              setAttributes();
  /// @brief build bin maps
  void                                              buildBinMaps();
  /// @brief summarize the statistics for movable and fixed instances
  void                                              setMovableFixedAreas();
  /// @brief init clock related data structures. net to clock mappings, clock region
  /// mapping, etc.
  void                                              initClock();
  /// @brief Check if fixed instances are placed on compatible site resource types.
  void                                              checkFixedInstToSiteRrscTypeCompatibility() const;


  container::ObserverPtr<Database>                  db_;             ///< bind database
  PlaceParams                                       place_params_;   ///< placement parameters

  /// netlist information in flat storage
  container::FlatNestedVector<IndexType, IndexType> inst_pins_;   ///< pins of each inst
  std::vector<IndexType>                            pin2inst_;    ///< map pin to its parent inst
  container::FlatNestedVector<IndexType, IndexType> net_pins_;    ///< pins of each net
  std::vector<IndexType>                            pin2net_;     ///< map pin to its parent net

  /// instance attributes
  // std::vector<SizeType> model_sizes_;       ///< model sizes only for non-module models
  std::vector<SizeType>                             inst_sizes_;   ///< instance sizes, # of instances x # of area types
  std::vector<PlaceStatus>                          inst_place_status_;   ///< placement status
  std::vector<Point3DType>                          inst_locs_;           ///< instance locations
  std::vector<std::set<IndexType>>                  inst_neighbors_;      ///< instance neighbors

  /// pin attributes
  std::vector<Point2DType>                          pin_offsets_;          ///< offset of pins to inst origin
  std::vector<SignalDirection>                      pin_signal_directs_;   ///< signal directions of pins
  std::vector<SignalType>                           pin_signal_types_;     ///< signal types of pins

  /// net attributes
  std::vector<WeightType>                           net_weights_;   ///< net weights

  /// layout information in flat storage
  BoxType                                           diearea_;        ///< layout area
  std::vector<DimType>                              bin_map_dims_;   ///< # of area types
                                        ///< dimensions of bin maps for different resource types
  std::vector<SizeType>                             bin_map_sizes_;   ///< # of area types
                                          ///< sizes of bin maps for different resource types
  std::vector<std::vector<CoordinateType>>          bin_capacity_maps_;   ///< # of area types, record capacity maps
                                                                          ///< for different resource types

  ///// map resource name to recorded resource types
  ///// this is different from the original resources read from the benchmarks
  // ResourceTypeEnum resource_type_enum_;

  std::vector<std::vector<uint8_t>>                 inst_resource_types_;   ///< resource types

  ///// map resource name to area type for global placement
  // AreaTypeEnum area_type_enum_;

  std::vector<std::vector<uint8_t>>                 inst_area_types_;   ///< the area types an inst can be placed in
  std::vector<std::vector<IndexType>>               area_type_inst_groups;   ///< instances grouped by area types

  std::vector<CoordinateType>                       total_movable_inst_areas_;   ///< # of area types,
                                                                                 ///< total movable
                                                                                 ///< instance areas
  std::vector<CoordinateType>                total_fixed_inst_areas_;   ///< # of area types, total fixed instance areas

  /// sort instances according to placement status
  std::vector<IndexType>                     new2old_inst_ids_;   ///< map from new id to old id
  std::vector<IndexType>                     old2new_inst_ids_;   ///< map from old id to new id
  std::pair<IndexType, IndexType>            movable_range_;      ///< range of movable instance indices
  std::pair<IndexType, IndexType>            fixed_range_;        ///< range of fixed instance indices

  /// statistics of resources and sites for building the mapping between
  /// resources and area types
  std::vector<IndexType>                     resource_num_sites_;        ///< # of resources, statistics
  std::vector<SizeType>                      resource_avg_site_sizes_;   ///< # of resources, statistics
  std::vector<SizeType>                      resource_max_site_sizes_;   ///< # of resources, statistics

  std::vector<IndexType>                     clock_nets_id2net_id_;   //

  IndexType                                  num_clock_nets_;

  std::vector<uint8_t>                       is_inst_luts_;
  std::vector<uint8_t>                       is_inst_ffs_;
  std::vector<uint8_t>                       is_inst_dsps_;
  std::vector<uint8_t>                       is_inst_rams_;
  std::vector<uint8_t>                       is_inst_clocksource_;

  std::vector<IndexType>                     clock_net_indexes_;
  std::vector<IndexType>                     clock_id_to_clock_net_id_;
  std::vector<IndexType>                     net_to_clocksource_;
  std::vector<std::vector<IndexType>>        inst_clock_indexes_;

  LayoutMap2D<ClockRegionRefType>            layout_to_clock_region_map;
  LayoutMap2D<HalfColumnRegionRefType>       layout_to_half_column_map;

  std::unordered_map<std::string, IndexType> name_to_inst;
  std::unordered_map<std::string, IndexType> name_to_net;

  // =============================================
  // [BEGIN ] >>>>>>> CarryChain Attribute >>>>>>>
  // =============================================

  container::FlatNestedVector<IndexType, IndexType> chain_insts_;   ///< the new inst Id of each chain

  // =============================================
  // [  END] <<<<<<<< CarryChain Attribute <<<<<<<
  // =============================================
};

}   // namespace database

OPENPARF_END_NAMESPACE

#endif
