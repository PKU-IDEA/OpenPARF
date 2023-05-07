/**
 * @file   database.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/database.h"

// c++ headers
#include <limits>
#include <unordered_map>
#include <vector>

// 3rd party library headers
#include <yaml-cpp/yaml.h>

#include <pugixml.hpp>

// project headers
#include "io/bookshelf/bookshelf_driver.h"
#include "io/verilog/VerilogDriver.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief A callback class to read bookshelf format
class BookshelfDatabaseCallbacks : public bookshelfparser::BookshelfDatabase {
 public:
  using IndexType = Object::IndexType;

  /// @brief constructor
  BookshelfDatabaseCallbacks(Database &db) : db_(db) {}
  /// @brief getter for input dir
  std::string const &inputDir() const { return input_dir_; }
  /// @brief getter for .aux file
  std::string const &auxFile() const { return aux_file_; }
  /// @brief setter for .aux file, convert to input directory and .aux file
  /// @param str is full path
  void               setAuxFile(std::string const &str) {
    auto pos = str.find_last_of('/');
    if (pos == std::string::npos) {
      input_dir_ = "./";
      aux_file_  = str;
    } else {
      input_dir_ = str.substr(0, pos + 1);
      aux_file_  = str.substr(pos + 1);
    }
  }
  /// @brief getter for .lib file
  std::string const &libFile() const { return lib_file_; }
  /// @brief getter for .scl file
  std::string const &sclFile() const { return scl_file_; }
  /// @brief getter for .node file
  std::string const &nodeFile() const { return node_file_; }
  /// @brief getter for .net file
  std::string const &netFile() const { return net_file_; }
  /// @brief getter for .pl file
  std::string const &plFile() const { return pl_file_; }
  /// @brief getter for .wt file
  std::string const &wtFile() const { return wt_file_; }
  /// @brief getter for .shape file
  std::string const &shapeFile() const { return shape_file_; }

  /* parsing aux file */
  void               setLibFileCbk(std::string const &str) { lib_file_ = str; }
  void               setSclFileCbk(std::string const &str) { scl_file_ = str; }
  void               setNodeFileCbk(std::string const &str) { node_file_ = str; }
  void               setNetFileCbk(std::string const &str) { net_file_ = str; }
  void               setPlFileCbk(std::string const &str) { pl_file_ = str; }
  void               setWtFileCbk(std::string const &str) { wt_file_ = str; }
  void               setShapeFileCbk(std::string const &str) { shape_file_ = str; }

  /* parsing lib file */
  void               addCellCbk(std::string const &str) {
    auto &design = db_.design();
    cur_model_   = &design.addModel(str);
    cur_model_->setModelType(ModelType::kCell);
  }
  void addCellInputPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(SignalType::kSignal);
  }
  void addCellOutputPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kOutput);
    mpin.setSignalType(SignalType::kSignal);
  }
  void addCellClockPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(SignalType::kClock);
  }
  void addCellCtrlPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(ctrlPinNameToPinType(str));
  }
  void addCellCtrlSRPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(SignalType::kControlSR);
  }
  void addCellCtrlCEPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(SignalType::kControlCE);
  }
  void addCellInputCasPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kInput);
    mpin.setSignalType(SignalType::kCascade);
  }
  void addCellOutputCasPinCbk(std::string const &str) {
    auto &mpin = cur_model_->tryAddModelPin(str);
    mpin.setSignalDirect(SignalDirection::kOutput);
    mpin.setSignalType(SignalType::kCascade);
  }
  void addCellParameterCbk(std::string const & /*str*/) {
    // parameters ignored for now
  }

  /* parsing scl file */
  void setSiteResources(std::string const                     &site_type,
          std::vector<std::pair<std::string, unsigned>> const &site_resources) {
    site_resources_.emplace_back(site_type, site_resources);
  }
  void addResourceTypeCbk(std::string const &resource_type, std::vector<std::string> const &cell_list) {
    for (auto const &cell : cell_list) {
      auto model_id = db_.design().modelId(cell);
      openparfAssertMsg(model_id < db_.design().numModels(), "model %s not defined", cell.c_str());
      if (db_.layout().resourceMap().numModels() != db_.design().numModels()) {
        db_.layout().resourceMap().setNumModels(db_.design().numModels());
      }
      db_.layout().resourceMap().tryAddModelResource(model_id, resource_type);
    }
  }
  void endResourceTypeBlockCbk() {
    auto &resource_map  = db_.layout().resourceMap();
    auto &site_type_map = db_.layout().siteTypeMap();
    for (auto const &kvp1 : site_resources_) {
      auto &site_type = site_type_map.tryAddSiteType(kvp1.first);
      site_type.setNumResources(resource_map.numResources());
      for (auto const &kvp2 : kvp1.second) {
        auto  resource_id = resource_map.resourceId(kvp2.first);
        auto &capacity    = kvp2.second;
        site_type.setResourceCapacity(resource_id, capacity);
      }
    }
    db_.layout().resizeNumSites(db_.layout().siteTypeMap().numSiteTypes());
    site_resources_.clear();
  }
  void initSiteMapCbk(unsigned w, unsigned h) { db_.layout().siteMap().reshape(w, h); }
  void endSiteMapCbk() {
    if (last_site_) {
      last_site_->setBbox({last_site_->siteMapId().x(), last_site_->siteMapId().y(), last_site_->siteMapId().x() + 1,
              last_site_->siteMapId().y() + db_.layout().siteMap().height() - last_site_->siteMapId().y()});
    }
  }
  /// @brief set the site dimension.
  /// The size of each site is derived from the index coordinates
  /// by assuming a column-based architecture.
  void setSiteMapEntryCbk(unsigned x, unsigned y, std::string const &site_type) {
    auto &site = db_.layout().addSite(x, y, db_.layout().siteTypeMap().siteTypeId(site_type));
    if (last_site_) {
      if (last_site_->siteMapId().x() == site.siteMapId().x()) {
        last_site_->setBbox({last_site_->siteMapId().x(), last_site_->siteMapId().y(), last_site_->siteMapId().x() + 1,
                last_site_->siteMapId().y() + site.siteMapId().y() - last_site_->siteMapId().y()});
      } else {
        last_site_->setBbox({last_site_->siteMapId().x(), last_site_->siteMapId().y(), last_site_->siteMapId().x() + 1,
                last_site_->siteMapId().y() + db_.layout().siteMap().height() - last_site_->siteMapId().y()});
      }
    }
    last_site_ = &site;
  }
  /// @brief set the site dimension.
  /// The size of each site is explictly given.
  void setSiteMapEntryWithBBoxCbk(unsigned xl, unsigned yl, unsigned xh, unsigned yh, std::string const &site_type) {
    auto &site = db_.layout().addSite(xl, yl, db_.layout().siteTypeMap().siteTypeId(site_type));
    site.setBbox({xl, yl, xh, yh});
    last_site_ = &site;
  }

  void initClockRegionsCbk(unsigned width, unsigned height) { db_.layout().clockRegionMap().reshape(width, height); }
  /// @brief add clock region.
  /// We assume the site width/height is 1, so the site indices are the same as
  /// coordindates.
  /// @param hc_xmin The left-most x boundary for the first half-column region
  /// in this clock region
  /// @param hc_ymid The y boundary for upper/lower half-column regions
  void addClockRegionCbk(std::string const &name,
          unsigned                          xl,
          unsigned                          yl,
          unsigned                          xh,
          unsigned                          yh,
          unsigned                          hc_ymid,
          unsigned                          hc_xmin) {
    openparfAssert(name.length() == 4 && name.substr(0, 1) == "X" && name.substr(2, 1) == "Y");
    using IndexType           = Object::IndexType;
    using CoordinateType      = Object::CoordinateType;

    IndexType clock_region_ix = std::stoi(name.substr(1, 1));
    IndexType clock_region_iy = std::stoi(name.substr(3, 1));
    auto     &clock_region    = db_.layout().clockRegionMap().at(clock_region_ix, clock_region_iy);
    openparfAssert(clock_region.id() == clock_region_ix * db_.layout().clockRegionMap().height() + clock_region_iy);
    clock_region.setName(name);
    clock_region.setBbox({(CoordinateType) xl, (CoordinateType) yl, (CoordinateType) xh, (CoordinateType) yh});
    clock_region.resizeNumSites(db_.layout().siteTypeMap().numSiteTypes());

    // set the clock region id in sites
    foreach2D(xl, yl, xh + 1, yh + 1, [&](IndexType ix, IndexType iy) {
      auto site_proxy = db_.layout().siteMap().at(ix, iy);
      if (site_proxy) {
        auto &site = *site_proxy;
        site.setClockRegionId(clock_region.id());
        clock_region.incrNumSites(site.siteTypeId(), 1);
      }
    });

    // add half column regions in the clock region
    // Iterate through all half-column regions in y-major order
    // So that each two half-column regions in the same columns are adjacent in
    // the _hcArray
    IndexType                                    hc_xmax = hc_xmin;
    std::array<std::array<CoordinateType, 2>, 2> y_ranges{};
    y_ranges[0] = {(CoordinateType) yl, (CoordinateType) hc_ymid - 1};
    y_ranges[1] = {(CoordinateType) hc_ymid, (CoordinateType) yh};
    for (IndexType hc_xl = hc_xmin; hc_xl <= xh; hc_xl += db_.params().arch_half_column_region_width_) {
      auto hc_xh = std::min(hc_xl + db_.params().arch_half_column_region_width_ - 1, xh);
      hc_xmax    = hc_xh;
      for (auto const &y_range : y_ranges) {
        auto  hc_yl = y_range[0], hc_yh = y_range[1];
        auto &hc = db_.layout().addHalfColumnRegion(clock_region.id());
        openparfAssert(hc.clockRegionId() == clock_region.id());
        openparfAssert(hc.id() == db_.layout().numHalfColumnRegions() - 1);
        hc.setBbox({(CoordinateType) hc_xl, hc_yl, (CoordinateType) hc_xh, hc_yh});
        // set half column region id in sites
        foreach2D<CoordinateType>(hc_xl, hc_yl, hc_xh, hc_yh, [&](IndexType ix, IndexType iy) {
          auto site_proxy = db_.layout().siteMap().at(ix, iy);
          if (site_proxy) {
            auto &site = *site_proxy;
            site.setHalfColumnRegionId(hc.id());
          }
        });
      }
    }
    clock_region.setHcXMin(hc_xmin);
    clock_region.setHcXMax(hc_xmax);
  }

  /// @brief add clock region without half column
  void addClockRegionWithoutHalfColumnCbk(unsigned xl,
          unsigned                                 yl,
          unsigned                                 xh,
          unsigned                                 yh,
          unsigned                                 clock_region_ix,
          unsigned                                 clock_region_iy) {
    using IndexType      = Object::IndexType;
    using CoordinateType = Object::CoordinateType;
    auto &clock_region   = db_.layout().clockRegionMap().at(clock_region_ix, clock_region_iy);
    openparfAssert(clock_region.id() == clock_region_ix * db_.layout().clockRegionMap().height() + clock_region_iy);
    clock_region.setName(std::to_string(clock_region_ix) + "_" + std::to_string(clock_region_iy));
    clock_region.setBbox({(CoordinateType) xl, (CoordinateType) yl, (CoordinateType) xh, (CoordinateType) yh});
    clock_region.resizeNumSites(db_.layout().siteTypeMap().numSiteTypes());
    // set the clock region id in sites
    foreach2D(xl, yl, xh + 1, yh + 1, [&](IndexType ix, IndexType iy) {
      auto site_proxy = db_.layout().siteMap().at(ix, iy);
      if (site_proxy) {
        auto &site = *site_proxy;
        site.setClockRegionId(clock_region.id());
        clock_region.incrNumSites(site.siteTypeId(), 1);
      }
    });
  }


  /* parsing nodes file */
  void addNodeCbk(std::string const &node_name, std::string const &cell_name) {
    if (!top_model_) {
      top_model_ = &db_.design().addModel("Bookshelf.TOP");
      top_model_->setModelType(ModelType::kModule);
      db_.design().setTopModuleInstId(top_model_->id());
    }
    auto model_id = db_.design().modelId(cell_name);
    if (model_id < db_.design().numModels()) {
      auto &model = db_.design().model(model_id);
      auto &inst  = top_model_->tryAddInst(node_name);
      if (inst.id() + 1 != top_model_->netlist()->numInsts()) {   // not newly added
        openparfPrint(kWarn, "inst %s with model %s already defined in model %s\n", node_name.c_str(),
                cell_name.c_str(), top_model_->name().c_str());
      }
      inst.attr().setModelId(model.id());
      inst.attr().setName(node_name);
    } else {
      openparfPrint(kError, "model %s for instance %d not found, ignored\n", cell_name.c_str(), node_name.c_str());
    }
  }

  /* parsing pl file */
  void setFixedNodeCbk(std::string const &node_name, unsigned x, unsigned y, unsigned z) {
    auto inst_proxy = top_model_->inst(node_name);
    if (inst_proxy) {
      auto &inst = *inst_proxy;
      inst.attr().setLoc({x, y, z});
      inst.attr().setPlaceStatus(PlaceStatus::kFixed);
    }
  }

  /* parsing nets file */
  void addNetCbk(std::string const &net_name, unsigned degree) {
    auto &net = top_model_->tryAddNet(net_name);
    if (net.id() + 1 != top_model_->netlist()->numNets()) {   // not newly added
      openparfPrint(kWarn, "net %s with degree %u already defined in model %s\n", net_name.c_str(), degree,
              top_model_->name().c_str());
    }
    net.pinIds().reserve(degree);
    cur_net_ = &net;
  }
  void addPinCbk(std::string const &node_name, std::string const &cell_pin_name) {
    auto inst_proxy = top_model_->inst(node_name);
    if (!inst_proxy) {
      openparfPrint(kWarn, "net %s instance %s not defined, ignored\n",
              top_model_->netlist()->net(cur_net_->id()).attr().name().c_str(), node_name.c_str());
      return;
    }
    auto       &inst    = *inst_proxy;
    auto const &attr    = top_model_->netlist()->inst(inst.id()).attr();
    auto const &model   = db_.design().model(attr.modelId());
    auto        mpin_id = model.modelPinId(cell_pin_name);
    if (mpin_id == std::numeric_limits<decltype(mpin_id)>::max()) {
      openparfPrint(kWarn, "net %s instance %s model pin %s not in model %s, ignored\n",
              top_model_->netlist()->net(cur_net_->id()).attr().name().c_str(), node_name.c_str(),
              cell_pin_name.c_str(), model.name().c_str());
    } else {
      auto &pin = top_model_->addPin();
      pin.setNetId(cur_net_->id());
      pin.setInstId(inst.id());
      inst.addPin(pin.id());
      cur_net_->addPin(pin.id());
      pin.setModelPinId(mpin_id);
      auto const &mpin = model.modelPin(mpin_id);
      pin.attr().setSignalDirect(mpin.signalDirect());
    }
  }

  /* parsing shape file */
  void addShapeCbk(std::string const &str) {
    ShapeConstrType t;
    if (str == "DSP-chain") {
      t = ShapeConstrType::kDSPChain;
    } else if (str == "RAM-chain") {
      t = ShapeConstrType::kRAMChain;
    } else if (str == "Carry-chain") {
      t = ShapeConstrType::kCarryChain;
    } else {
      t = ShapeConstrType::kUnknown;
      openparfAssertMsg(t != ShapeConstrType::kUnknown, "unknown shape constraint type %s", str.c_str());
    }
    cur_shape_constr_ = &db_.design().addShapeConstr(t);
  }
  void addShapeNodeCbk(unsigned x, unsigned y, std::string const &node_name) {
    auto inst_id = top_model_->instId(node_name);
    if (inst_id == std::numeric_limits<decltype(inst_id)>::max()) {
      openparfPrint(kWarn, "shape constraint %u element %u instance %s not found, ignored\n", cur_shape_constr_->id(),
              cur_shape_constr_->numShapeElements(), node_name.c_str());
    }
    cur_shape_constr_->addShapeElement({x, y}, inst_id);
  }

 protected:
  Database    &db_;   ///< reference to a Database object

  /// bookshelf files and path
  std::string  input_dir_;    ///< input file directory
  std::string  aux_file_;     ///< base .aux file name
  std::string  lib_file_;     ///< base .lib file name
  std::string  scl_file_;     ///< base .scl file name
  std::string  node_file_;    ///< base .node file name
  std::string  pl_file_;      ///< base .pl file name
  std::string  net_file_;     ///< base .net file name
  std::string  wt_file_;      ///< base .wt file name
  std::string  shape_file_;   ///< base .shape file name

  /// temporary storage during parsing
  Model       *top_model_        = nullptr;   ///< assume Bookshelf file does not contain any hierarcy
  Model       *cur_model_        = nullptr;   ///< current model
  Net         *cur_net_          = nullptr;   ///< current net
  ShapeConstr *cur_shape_constr_ = nullptr;   ///< current shape constraint
  Site        *last_site_        = nullptr;   ///< record last site
  std::vector<std::pair<std::string, std::vector<std::pair<std::string, IndexType>>>>
          site_resources_;   ///< temporary stoarge of site resources,
                             ///< because we need to wait until resource map
                             ///< constructed (site type, vector of (resource type,
                             ///< capacity))
};

class VerilogDatabaseCallbacks : public VerilogParser::VerilogDataBase {
 public:
  VerilogDatabaseCallbacks(Database &db) : db_(db) {
    top_model_ = &db_.design().addModel("Verilog.TOP");
    top_model_->setModelType(ModelType::kModule);
    db_.design().setTopModuleInstId(top_model_->id());
  }
  /// @brief read a module declaration
  ///
  /// module NOR2_X1 ( a, b, c );
  ///
  /// @param module_name name of a module
  /// @param vPinName array of pins
  void verilog_module_declaration_cbk(std::string const &module_name,
          std::vector<VerilogParser::GeneralName> const &vPinName) {
    // This declares the top level module. We needn't do anything.
    return;
  }

  /// @brief read an instance.
  ///
  /// NOR2_X1 u2 ( .a(n1), .b(n3), .o(n2) );
  /// NOR2_X1 u2 ( .a(n1), .b({n3, n4}), .o(n2) );
  /// NOR2_X1 u2 ( .a(n1), .b(1'b0), .o(n2) );
  ///
  /// @param macro_name standard cell type or module name
  /// @param inst_name instance name
  /// @param vNetPin array of pairs of net and pin
  void verilog_instance_cbk(std::string const      &macro_name,
          std::string const                        &inst_name,
          std::vector<VerilogParser::NetPin> const &vNetPin) {
    if (inst_name == "BUFG_inst_7_inst_9_ClockGenerator/BUFG_BUFG_config6_inst110_cpy_27") {
      int a;
      a++;
    }
    if (macro_name == "GLOBAL_INPAD" || macro_name == "GCLK_BUF" || macro_name == "RCLK_BUF") {
      openparfPrint(kWarn, "Clock model %s is not finished.\n", macro_name.c_str());
    }
    auto model_id = db_.design().modelId(macro_name);
    if (model_id < db_.design().numModels()) {
      auto &model = db_.design().model(model_id);
      auto &inst  = top_model_->tryAddInst(inst_name);
      if (inst.id() + 1 != top_model_->netlist()->numInsts()) {   // not newly added
        openparfPrint(kWarn, "inst %s with model %s already defined in model %s\n", inst_name.c_str(),
                macro_name.c_str(), top_model_->name().c_str());
      }
      inst.attr().setModelId(model.id());
      inst.attr().setName(inst_name);
    } else {
      openparfPrint(kError, "model %s for instance %d not found, ignored\n", macro_name.c_str(), inst_name.c_str());
    }
    auto       &inst  = *top_model_->inst(inst_name);
    auto const &attr  = top_model_->netlist()->inst(inst.id()).attr();
    auto const &model = db_.design().model(attr.modelId());
    for (VerilogParser::NetPin const &netpin : vNetPin) {
      std::string pin_name = netpin.pin;
      auto        mpin_id  = model.modelPinId(netpin.pin);
      auto        net      = top_model_->net(netpin.net);
      if (netpin.net == "VerilogParser::GROUP_NETS") {
        int32_t pin_bit_size = netpin.extension.vNetName->size();
        for (int i = 0; i < pin_bit_size; i++) {
          auto        new_net       = top_model_->net(netpin.extension.vNetName->at(i).name);
          // The bit order of group pins is {IN[3], IN[2], IN[1], IN[0]}
          std::string curr_pin_name = pin_name + "[" + std::to_string(pin_bit_size - 1 - i) + "]";
          auto        curr_mpin_id  = model.modelPinId(curr_pin_name);
          auto       &pin           = top_model_->addPin();
          pin.setNetId(new_net->id());
          pin.setInstId(inst.id());
          inst.addPin(pin.id());
          new_net->addPin(pin.id());
          pin.setModelPinId(curr_mpin_id);
          auto const &curr_mpin = model.modelPin(curr_mpin_id);
          pin.attr().setSignalDirect(curr_mpin.signalDirect());
        }
      } else {
        if (mpin_id == std::numeric_limits<decltype(mpin_id)>::max()) {
          // Try pin_name + "[0]"
          pin_name += "[" + std::to_string(0) + "]";
          mpin_id = model.modelPinId(pin_name);
          openparfAssertMsg(mpin_id != std::numeric_limits<decltype(mpin_id)>::max(), "%s does not exist\n",
                  pin_name.c_str());
        }
        auto &pin = top_model_->addPin();
        pin.setNetId(net->id());
        pin.setInstId(inst.id());
        inst.addPin(pin.id());
        net->addPin(pin.id());
        pin.setModelPinId(mpin_id);
        auto const &curr_mpin = model.modelPin(mpin_id);
        pin.attr().setSignalDirect(curr_mpin.signalDirect());
      }
    }
  }
  /// @brief read an net declaration
  ///
  /// wire aaa[1];
  ///
  /// @param net_name net name
  /// @param range net range, negative infinity if either low or high value of the range is not defined
  void verilog_net_declare_cbk(std::string const &net_name, VerilogParser::Range const &range) {
    openparfAssert(range.low == std::numeric_limits<int>::min() && range.high == std::numeric_limits<int>::min());
    auto &net = top_model_->tryAddNet(net_name);
    if (net.id() + 1 != top_model_->netlist()->numNets()) {   // not newly added
      openparfPrint(kWarn, "net %s already defined in model %s\n", net_name.c_str(), top_model_->name().c_str());
    }
  }
  /// @brief read an pin declaration
  ///
  /// input inp2;
  ///
  /// @param pin_name pin name
  /// @param type type of pin, refer to @ref VerilogParser::PinType
  /// @param range pin range, negative infinity if either low or high value of the range is not defined
  void verilog_pin_declare_cbk(std::string const &pin_name, unsigned type, VerilogParser::Range const &range) {
    openparfAssert(range.low == std::numeric_limits<int>::min() && range.high == std::numeric_limits<int>::min());
    openparfAssert(type == VerilogParser::kINPUT || type == VerilogParser::kOUTPUT);
    auto &top_model_pin = top_model_->tryAddModelPin(pin_name);
    if (type == VerilogParser::kINPUT) {
      top_model_pin.setSignalDirect(SignalDirection::kInput);
    } else if (type == VerilogParser::kOUTPUT) {
      top_model_pin.setSignalDirect(SignalDirection::kOutput);
    }
    // Make a special net
    std::string net_name = pin_name;
    auto       &net      = top_model_->tryAddNet(net_name);
    top_model_->setModelPinNet(top_model_pin.id(), net.id());
  }
  /// @brief read an assignment
  void verilog_assignment_cbk(std::string const &target_name,
          VerilogParser::Range const            &target_range,
          std::string const                     &source_name,
          VerilogParser::Range const            &source_range) {
    openparfPrint(kError, "Assignment should not happen for xarch benchmark\n");
  }

  void set_fixed_node_cbk(std::string const &node_name, unsigned x, unsigned y, unsigned z) {
    auto inst_proxy = top_model_->inst(node_name);
    if (inst_proxy) {
      auto &inst = *inst_proxy;
      inst.attr().setLoc({x, y, z});
      inst.attr().setPlaceStatus(PlaceStatus::kFixed);
    }
  }

 private:
  Database &db_;                    ///< reference to a Database object
  Model    *top_model_ = nullptr;   ///< assume Verilog file does not contain any hierarcy
  Model    *cur_model_ = nullptr;   ///< current model
  Net      *cur_net_   = nullptr;   ///< current net
};

void Database::readXML(std::string const &xml_file) {
  pugi::xml_document     doc;
  pugi::xml_parse_result result       = doc.load_file(xml_file.c_str());
  pugi::xml_node         arch         = doc.document_element();
  pugi::xml_node         core         = arch.child("core");
  auto                   StrBeginWith = [](const std::string &a, const std::string &b) { return a.rfind(b, 0) == 0; };
  // We use the bookshelf parsers's callbacks
  BookshelfDatabaseCallbacks cbk(*this);
  // Add cells/models/primitives
  openparfPrint(kDebug, "XML Parser: Adding primitives:\n");
  int32_t num_primitive = 0;
  for (pugi::xml_node primitive : arch.child("primitives").children("primitive")) {
    num_primitive++;
    std::string primitive_name = primitive.attribute("name").as_string();
    cbk.addCellCbk(primitive_name);
    openparfPrint(kDebug, "    Model %s added.\n", primitive.attribute("name").value());
    for (pugi::xml_node pin : primitive.children()) {
      for (int cnt = 0; cnt < pin.attribute("width").as_int(); cnt++) {
        std::string pin_name = pin.attribute("name").as_string();
        if (pin.attribute("width").as_int() > 1) {
          pin_name += "[" + std::to_string(cnt) + "]";
        }
        std::string pin_type = pin.name();
        std::string pin_attr = pin.attribute("attr").value();
        if (pin_type == "input" && pin_attr == "enable") {
          cbk.addCellCtrlCEPinCbk(pin_name);
        } else if (pin_type == "input" && pin_attr == "clock") {
          cbk.addCellClockPinCbk(pin_name);
        } else if (pin_type == "input" && pin_attr == "reset") {
          cbk.addCellCtrlSRPinCbk(pin_name);
        } else if (primitive_name == "CLA4" && StrBeginWith(pin_name, "CAS_IN")) {
          cbk.addCellInputCasPinCbk(pin_name);
        } else if (primitive_name == "CLA4" && StrBeginWith(pin_name, "CAS_OUT")) {
          cbk.addCellOutputCasPinCbk(pin_name);
        } else if (primitive_name == "GCU0" && StrBeginWith(pin_name, "PCIN")) {
          cbk.addCellInputCasPinCbk(pin_name);
        } else if (primitive_name == "GCU0" && StrBeginWith(pin_name, "PCOUT")) {
          cbk.addCellOutputCasPinCbk(pin_name);
        } else if (pin_type == "input") {
          cbk.addCellInputPinCbk(pin_name);
        } else if (pin_type == "output") {
          cbk.addCellOutputPinCbk(pin_name);
        } else {
          openparfAssertMsg(false, "Unknown pin type %s\n", pin_type);
        }
      }
    }
  }
  openparfPrint(kDebug, "Added %i primitives \n", num_primitive);

  // Add sites types and resources
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> tile_size;
  for (pugi::xml_node tile : arch.child("tile_blocks").children("tile")) {
    int32_t     width  = tile.attribute("width").as_int();
    int32_t     height = tile.attribute("height").as_int();
    std::string type   = tile.attribute("type").value();
    tile_size.emplace(type, std::make_pair(width, height));
    openparfPrint(kDebug, "Tile %s has a size (%i, %i)\n", type.c_str(), width, height);
  }

  // TODO(Jing Mai): Even if we have to gen this manually, can we just get it from a config file?
  cbk.setSiteResources("DUMMY", {});
  cbk.setSiteResources("IO", {std::make_pair("INPAD", 24), std::make_pair("OUTPAD", 24)});
  cbk.setSiteResources("RAMAT", {std::make_pair("RAMA", 1)});
  cbk.setSiteResources("RAMBT", {std::make_pair("RAMB", 1)});
  cbk.setSiteResources("RAMCT", {std::make_pair("RAMC", 1)});
  cbk.setSiteResources("FUAT", {std::make_pair("CLA4", 2), std::make_pair("DFF", 16), std::make_pair("LUTL", 16)});
  cbk.setSiteResources("FUBT", {std::make_pair("CLA4", 2), std::make_pair("DFF", 16), std::make_pair("LUTM", 16)});

  cbk.addResourceTypeCbk("INPAD", {"INPAD"});
  cbk.addResourceTypeCbk("OUTPAD", {"OUTPAD"});
  cbk.addResourceTypeCbk("RAMA", {"BRAM36K"});
  cbk.addResourceTypeCbk("RAMB", {"RAMB"});
  cbk.addResourceTypeCbk("RAMC", {"GCU0"});
  cbk.addResourceTypeCbk("CLA4", {"CLA4"});
  cbk.addResourceTypeCbk("DFF", {"DFF"});
  cbk.addResourceTypeCbk("LUTL", {"LUT5", "LUT6"});
  cbk.addResourceTypeCbk("LUTM", {"LUT5", "LUT6", "SHIFT", "LRAM"});
  cbk.endResourceTypeBlockCbk();

  for (pugi::xml_node grid = core.child("grid"); grid; grid = grid.next_sibling("grid")) {
    std::string t = grid.attribute("type").value();
    if (t == "CLOCK_GRID") {
      // Currently, CLOCK_GRID's attributes are the same as TILE_GRID
      // This might change in the future
      // Currently, we needn't do anything
    } else if (t == "TILE") {
      int32_t     width        = grid.attribute("width").as_int();
      int32_t     height       = grid.attribute("height").as_int();
      std::string default_type = grid.child("default").attribute("type").value();
      auto const &site_map     = layout().siteMap();

      cbk.initSiteMapCbk(width, height);

      // TODO(Jing Mai): correct implementation of priority
      for (pugi::xml_node inst : grid.children("inst")) {
        std::string type = inst.attribute("type").value();
        // It does not make sense to place inst tiles other than DUMMY
        openparfAssert(type == "DUMMY");
        int32_t dx      = tile_size.at(type).first;
        int32_t dy      = tile_size.at(type).second;
        int32_t start_x = inst.attribute("start_x").as_int();
        int32_t start_y = inst.attribute("start_y").as_int();
        cbk.setSiteMapEntryWithBBoxCbk(start_x, start_y, start_x + dx, start_y + dy, type);
      }

      for (pugi::xml_node region : grid.children("region")) {
        std::string type    = region.attribute("type").value();
        int32_t     dx      = tile_size.at(type).first;
        int32_t     dy      = tile_size.at(type).second;
        int32_t     start_x = region.attribute("start_x").as_int();
        int32_t     end_x   = region.attribute("end_x").as_int();
        int32_t     start_y = region.attribute("start_y").as_int();
        int32_t     end_y   = region.attribute("end_y").as_int();
        for (uint32_t x = start_x; x <= end_x; x += dx)
          for (uint32_t y = start_y; y <= end_y; y += dy) {
            cbk.setSiteMapEntryWithBBoxCbk(x, y, x + dx, y + dy, type);
          }
      }
      /* Now add default tiles */ {
        int32_t dx = tile_size.at(default_type).first;
        int32_t dy = tile_size.at(default_type).second;
        for (uint32_t x = 0; x < width; x += dx) {
          for (uint32_t y = 0; y < height; y += dy) {
            if (site_map.at(x, y)) continue;
            if (site_map.overlapAt(x, y)) continue;
            cbk.setSiteMapEntryWithBBoxCbk(x, y, x + dx, y + dy, default_type);
          }
        }
      }
    }
  }

  pugi::xml_node cr = core.child("clock_region");
  cbk.initClockRegionsCbk(cr.attribute("width").as_int(), cr.attribute("height").as_int());
  for (pugi::xml_node cr_region : cr.children("region")) {
    unsigned xl  = cr_region.attribute("start_x").as_uint();
    unsigned xh  = cr_region.attribute("end_x").as_uint();
    unsigned yl  = cr_region.attribute("start_y").as_uint();
    unsigned yh  = cr_region.attribute("end_y").as_uint();
    unsigned crx = cr_region.attribute("cr_x").as_uint();
    unsigned cry = cr_region.attribute("cr_y").as_uint();
    cbk.addClockRegionWithoutHalfColumnCbk(xl, yl, xh, yh, crx, cry);
  }
  openparfPrint(kDebug, "Done parsing\n");
}

void Database::readBookshelf(std::string const &aux_file) {
  BookshelfDatabaseCallbacks cbk(*this);
  cbk.setAuxFile(aux_file);
  bookshelfparser::Driver driver(cbk);
  std::ifstream           ifs;

  // Parse bookshelf AUX file
  openparfPrint(kInfo, "Parsing file %s\n", aux_file.c_str());
  ifs.open(aux_file);
  openparfAssertMsg(ifs.good(), "Cannot open file %s", aux_file.c_str());
  driver.parse_stream(ifs);
  ifs.close();

  // Parse bookshelf files in order
  for (auto file :
          {cbk.libFile(), cbk.sclFile(), cbk.nodeFile(), cbk.plFile(), cbk.netFile(), cbk.shapeFile(), cbk.wtFile()}) {
    if (!file.empty()) {
      std::string path = cbk.inputDir() + file;
      openparfPrint(kInfo, "Parsing file %s\n", path.c_str());
      ifs.open(path);
      openparfAssertMsg(ifs.good(), "Cannot open file %s", path.c_str());
      driver.parse_stream(ifs);
      ifs.close();
    }
  }

  // uniquify the design
  design_.uniquify(design_.topModuleInstId());
}

bool Database::writeBookshelfPl(std::string const &pl_file) {
  openparfPrint(kInfo, "write to %s\n", pl_file.c_str());
  std::ofstream ofs(pl_file.c_str());
  if (!ofs.good()) {
    openparfPrint(kError, "failed to open file %s for write", pl_file.c_str());
    return false;
  }

  // valid site map
  auto const            &site_map = layout().siteMap();
  std::vector<IndexType> valid_site_map(site_map.width() * site_map.height(), std::numeric_limits<IndexType>::max());
  for (auto const &site : site_map) {
    auto const &bbox = site.bbox();
    for (IndexType ix = bbox.xl(); ix < bbox.xh(); ++ix) {
      for (IndexType iy = bbox.yl(); iy < bbox.yh(); ++iy) {
        valid_site_map.at(ix * site_map.height() + iy) = site_map.index1D(site.siteMapId().x(), site.siteMapId().y());
      }
    }
  }

  auto        top_module_inst = design_.topModuleInst();
  // assume flat netlist for now
  auto const &netlist         = top_module_inst->netlist();
  for (auto const &inst_id : netlist.instIds()) {
    auto const &inst = netlist.inst(inst_id);
    IndexType   id1d = valid_site_map.at(inst.attr().loc().x() * site_map.height() + inst.attr().loc().y());
    openparfAssertMsg(id1d != std::numeric_limits<IndexType>::max(),
            "Instance %s(%d, %d) is not located in any vaild site.", inst.attr().name().c_str(), inst.attr().loc().x(),
            inst.attr().loc().y());
    auto const &site = site_map.at(id1d);
    ofs << inst.attr().name() << " " << site->bbox().xl() << " " << site->bbox().yl() << " " << inst.attr().loc().z();
    if (inst.attr().placeStatus() == PlaceStatus::kFixed) {
      ofs << " FIXED";
    }
    ofs << "\n";
  }

  ofs.close();
  return true;
}

bool Database::writeBookshelfNodes(std::string const &nodes_file) {
  auto          top_module_inst = design_.topModuleInst();
  auto         &netlist         = top_module_inst->netlist();
  std::ofstream ofs(nodes_file.c_str());

  DEFER({ ofs.close(); });
  openparfAssert(top_module_inst);
  openparfPrint(kInfo, "write to %s\n", nodes_file.c_str());
  if (!ofs.good()) {
    openparfPrint(kError, "failed to open file %s for write\n", nodes_file.c_str());
    return false;
  }

  for (auto const &inst_id : netlist.instIds()) {
    const auto        &inst       = netlist.inst(inst_id);
    int32_t            model_id   = inst.attr().modelId();
    const auto        &model      = design_.model(model_id);
    const std::string &inst_name  = inst.attr().name();
    const std::string &model_name = model.name();
    ofs << inst_name << " " << model_name << std::endl;
  }
  return true;
}

bool Database::writeBookshelfNets(std::string const &nets_file) {
  auto          top_module_inst = design_.topModuleInst();
  auto         &netlist         = top_module_inst->netlist();
  std::ofstream ofs(nets_file.c_str());

  DEFER({ ofs.close(); });
  openparfAssert(top_module_inst);
  openparfPrint(kInfo, "write to %s\n", nets_file.c_str());
  if (!ofs.good()) {
    openparfPrint(kError, "failed to open file %s for write\n", nets_file.c_str());
    return false;
  }
  for (auto const &net_id : netlist.netIds()) {
    const auto        &net      = netlist.net(net_id);
    const auto        &pin_ids  = net.pinIds();
    int32_t            net_size = pin_ids.size();
    const std::string &net_name = net.attr().name();
    std::string        indent   = "    ";
    ofs << "net " << net_name << " " << net_size << std::endl;
    for (const auto &pin_id : pin_ids) {
      const auto        &pin            = netlist.pin(pin_id);
      int32_t            inst_id        = pin.instId();
      const auto        &inst           = netlist.inst(inst_id);
      int32_t            model_id       = inst.attr().modelId();
      const auto        &model          = design_.model(model_id);
      int32_t            model_pin_id   = pin.modelPinId();
      const auto        &model_pin      = model.modelPin(model_pin_id);
      const std::string &inst_name      = inst.attr().name();
      const std::string &model_pin_name = model_pin.name();
      ofs << indent << inst_name << " " << model_pin_name << std::endl;
    }
    ofs << "endnet" << std::endl;
  }
  return true;
}

void Database::readFlexshelf(std::string const &layout_file,
        std::string const                      &netlist_file,
        std::string const                      &place_file) {
  readFlexshelfLayout(layout_file);
  readFlexshelfDesign(netlist_file, place_file);
}

void Database::readFlexshelfLayout(std::string const &layout_file) {
  // ==================== Read Layout ====================

  pugi::xml_document     doc;
  pugi::xml_parse_result result = doc.load_file(layout_file.c_str());
  if (!result) {
    openparfPrint(kError, "failed to open file %s for read\n", layout_file.c_str());
    return;
  }
  pugi::xml_node             arch = doc.child("arch");
  pugi::xml_node             core = arch.child("core");
  BookshelfDatabaseCallbacks cbk(*this);
  openparfPrint(kInfo, "read layout from %s\n", layout_file.c_str());

  // ------------------- read sites -------------------
  struct SiteSize {
    int32_t width;
    int32_t height;
    int32_t offset_x;
    int32_t offset_y;
  };

  int32_t                                                      num_tile = 0;
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> tile_sizes;
  std::unordered_map<std::string, std::vector<SiteSize>>       site_sizes;
  for (pugi::xml_node tile : arch.child("tile_blocks").children("tile")) {
    std::string type        = tile.attribute("type").as_string();
    int32_t     tile_width  = tile.attribute("width").as_int();
    int32_t     tile_height = tile.attribute("height").as_int();
    num_tile += 1;
    tile_sizes[type] = {tile_width, tile_height};

    if (tile.children("sub_tiles").empty()) {
      // read site size
      site_sizes[type].push_back({tile_width, tile_height, 0, 0});
      openparfPrint(kInfo, "tile %s size: %d %d\n", type.c_str(), tile_width, tile_height);

      // read resource capacity
      std::vector<std::pair<std::string, uint32_t>> resource_caps;
      for (pugi::xml_node resource : tile.children("resource")) {
        std::string resource_name = resource.attribute("name").as_string();
        int32_t     cap           = resource.attribute("num").as_int();
        resource_caps.emplace_back(resource_name, cap);
        openparfPrint(kInfo, "tile %s resource %s capacity %d\n", type.c_str(), resource_name.c_str(), cap);
      }
      cbk.setSiteResources(type, resource_caps);
    } else {
      // read sub tile size
      for (pugi::xml_node sub_tile : tile.child("sub_tiles").children("sub_tile")) {
        int32_t width    = sub_tile.attribute("width").as_int();
        int32_t height   = sub_tile.attribute("height").as_int();
        int32_t offset_x = sub_tile.attribute("offset_x").as_int();
        int32_t offset_y = sub_tile.attribute("offset_y").as_int();
        site_sizes[type].push_back({width, height, offset_x, offset_y});
        openparfPrint(kInfo, "tile %s sub site size: %d %d offset: %d %d\n", type.c_str(), width, height, offset_x,
                offset_y);
      }

      // read resource capacity
      std::vector<std::pair<std::string, uint32_t>> resource_caps;
      for (pugi::xml_node resource : tile.child("sub_tiles").children("resource")) {
        std::string resource_name = resource.attribute("name").as_string();
        int32_t     cap           = resource.attribute("num").as_int();
        resource_caps.emplace_back(resource_name, cap);
        openparfPrint(kInfo, "site %s resource %s capacity %d\n", type.c_str(), resource_name.c_str(), cap);
      }
      cbk.setSiteResources(type, resource_caps);
    }
  }
  openparfPrint(kInfo, "read %d tiles\n", num_tile);

  // ------------------- read primitives -------------------
  int32_t num_primitive = 0;
  for (pugi::xml_node primitive : arch.child("primitives").children("primitive")) {
    num_primitive++;
    std::string primitive_name = primitive.attribute("name").as_string();
    cbk.addCellCbk(primitive_name);
    openparfPrint(kDebug, "primitive %s\n", primitive_name.c_str());
    for (pugi::xml_node pin : primitive.children()) {
      int32_t width = pin.attribute("width").as_int();
      for (int32_t cnt = 0; cnt < width; cnt++) {
        std::string pin_direction = pin.name();
        std::string pin_type      = pin.attribute("type").as_string();
        std::string pin_name      = pin.attribute("name").as_string();
        if (width > 1) {
          // TODO: support multi-bit pin
          pin_name += "[" + std::to_string(cnt) + "]";
        }
        if (pin_direction == "input" && pin_type == "ENABLE") {
          cbk.addCellCtrlCEPinCbk(pin_name);
        } else if (pin_direction == "input" && pin_type == "CLOCK") {
          cbk.addCellClockPinCbk(pin_name);
        } else if (pin_direction == "input" && pin_type == "RESET") {
          cbk.addCellCtrlSRPinCbk(pin_name);
        } else if (pin_direction == "input" && pin_type == "CTRL") {
          cbk.addCellCtrlPinCbk(pin_name);
        } else if (pin_direction == "input") {
          cbk.addCellInputPinCbk(pin_name);
        } else if (pin_direction == "output") {
          cbk.addCellOutputPinCbk(pin_name);
        } else {
          openparfPrint(kError, "unknown pin type %s %s", pin_direction.c_str(), pin_type.c_str());
        }
      }
    }
    // resource demand
    for (pugi::xml_node resource : primitive.children("resource")) {
      std::string resource_name = resource.attribute("name").as_string();
      int32_t     demand        = resource.attribute("num").as_int();
      cbk.addResourceTypeCbk(resource_name, {primitive_name});
      openparfPrint(kInfo, "primitive %s resource %s demand %d\n", primitive_name.c_str(), resource_name.c_str(),
              demand);
    }
  }
  openparfPrint(kInfo, "read %d primitives\n", num_primitive);
  cbk.endResourceTypeBlockCbk();

  // ------------------- read tile grid -------------------
  int32_t                                  num_sites  = 0;
  bool                                     grid_found = false;
  std::unordered_map<std::string, int32_t> site_counts;
  for (pugi::xml_node grid : core.children("grid")) {
    std::string grid_name = grid.attribute("name").as_string();
    if (grid_name == "TILE_GRID") {
      grid_found               = true;
      int32_t     width        = grid.attribute("width").as_int();
      int32_t     height       = grid.attribute("height").as_int();
      std::string default_type = grid.child("default").attribute("type").as_string();
      auto const &site_map     = layout().siteMap();

      cbk.initSiteMapCbk(width, height);

      for (pugi::xml_node inst : grid.children("region")) {
        std::string type    = inst.attribute("type").as_string();
        int32_t     start_x = inst.attribute("start_x").as_int();
        int32_t     start_y = inst.attribute("start_y").as_int();
        int32_t     end_x   = inst.attribute("end_x").as_int();
        int32_t     end_y   = inst.attribute("end_y").as_int();

        openparfAssertMsg(tile_sizes.find(type) != tile_sizes.end(), "unknown tile type %s", type.c_str());
        openparfAssertMsg(site_sizes.find(type) != site_sizes.end(), "unknown site type %s", type.c_str());
        for (int32_t tx = start_x; tx <= end_x; tx += tile_sizes[type].first) {
          for (int32_t ty = start_y; ty <= end_y; ty += tile_sizes[type].second) {
            for (SiteSize site_size : site_sizes[type]) {
              num_sites += 1;
              site_counts[type] += 1;
              cbk.setSiteMapEntryWithBBoxCbk(tx + site_size.offset_x, ty + site_size.offset_y,
                      tx + site_size.offset_x + site_size.width, ty + site_size.offset_y + site_size.height, type);
            }
          }
        }
      }

      // now add default tile
      int32_t dx = tile_sizes[default_type].first;
      int32_t dy = tile_sizes[default_type].second;
      openparfAssertMsg(site_sizes[default_type].size() == 1, "default tile should have only one site");
      openparfAssertMsg(dx == 1 && dy == 1, "default tile should be 1x1");
      for (int32_t x = 0; x < width; x++) {
        for (int32_t y = 0; y < height; y++) {
          if (!site_map.at(x, y) && !site_map.overlapAt(x, y)) {
            cbk.setSiteMapEntryWithBBoxCbk(x, y, x + dx, y + dy, default_type);
            num_sites += 1;
            site_counts[default_type] += 1;
          }
        }
      }
      openparfPrint(kInfo, "read %d sites\n", num_sites);
      for (auto const &site_count : site_counts) {
        openparfPrint(kInfo, "site type %s count %d\n", site_count.first.c_str(), site_count.second);
      }
    }
  }
  openparfAssertMsg(grid_found, "tile grid not found");

  // ------------------- read clock grid -------------------
  int32_t        num_crs    = 0;
  pugi::xml_node clock_grid = core.child("clock_region");
  int32_t        width      = clock_grid.attribute("width").as_int();
  int32_t        height     = clock_grid.attribute("height").as_int();
  cbk.initClockRegionsCbk(width, height);
  for (pugi::xml_node clock_region : clock_grid.children("region")) {
    num_crs += 1;
    std::string cr_name = clock_region.attribute("name").as_string();
    int32_t     start_x = clock_region.attribute("start_x").as_int();
    int32_t     start_y = clock_region.attribute("start_y").as_int();
    int32_t     end_x   = clock_region.attribute("end_x").as_int();
    int32_t     end_y   = clock_region.attribute("end_y").as_int();
    int32_t     hc_ymid = clock_region.attribute("hc_ymid").as_int();
    int32_t     hc_xmin = clock_region.attribute("hc_xmin").as_int();
    cbk.addClockRegionCbk(cr_name, start_x, start_y, end_x, end_y, hc_ymid, hc_xmin);
  }
  openparfPrint(kInfo, "read %d clock regions\n", num_crs);
}

void Database::readFlexshelfDesign(const std::string &netlist_file, const std::string &place_file) {
  // ------------------- read netlist -------------------
  openparfPrint(kInfo, "reading design %s\n", netlist_file.c_str());
  VerilogDatabaseCallbacks cbk(*this);
  VerilogParser::read(cbk, netlist_file);

  // ------------------- read placement -------------------
  openparfPrint(kInfo, "reading placement %s\n", place_file.c_str());
  pugi::xml_document     doc;
  pugi::xml_parse_result result = doc.load_file(place_file.c_str());

  if (!result) {
    openparfPrint(kError, "Error parsing xml file: %s\n", result.description());
  }
  for (pugi::xml_node cell : doc.child("design").child("position").children("cell")) {
    std::string name   = cell.attribute("name").as_string();
    int32_t     x      = cell.attribute("x").as_int();
    int32_t     y      = cell.attribute("y").as_int();
    int32_t     z      = cell.attribute("z").as_int();
    std::string status = cell.attribute("status").as_string();
    if (status == "FIXED") {
      cbk.set_fixed_node_cbk(name, x, y, z);
    } else {
      openparfPrint(kError, "unsupported status %s", status.c_str());
    }
  }

  // ------------------- uniquify -------------------
  design_.uniquify(design_.topModuleInstId());
}

void Database::readVerilog(std::string const &verilog_file) {
  VerilogDatabaseCallbacks cbk(*this);
  VerilogParser::read(cbk, verilog_file);
  design_.uniquify(design_.topModuleInstId());
}

std::ostream &operator<<(std::ostream &os, Database::Params const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", arch_half_column_region_width_ : " << rhs.arch_half_column_region_width_;
  os << ")";
  return os;
}

void Database::copy(Database const &rhs) {
  this->BaseType::copy(rhs);
  design_ = rhs.design_;
  layout_ = rhs.layout_;
  params_ = rhs.params_;
}

void Database::move(Database &&rhs) {
  this->BaseType::move(std::move(rhs));
  design_ = std::move(rhs.design_);
  layout_ = std::move(rhs.layout_);
  params_ = std::move(rhs.params_);
}

Database::IndexType Database::memory() const {
  IndexType ret = this->BaseType::memory() + +design_.memory() + layout_.memory() + sizeof(params_);
  return ret;
}

std::ostream &operator<<(std::ostream &os, Database const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", \ndesign_ : " << rhs.design_ << std::endl;
  os << ", \nlayout_ : " << rhs.layout_ << std::endl;
  os << ", \nparams_ : " << rhs.params_ << std::endl;
  os << ")";
  return os;
}

}   // namespace database

OPENPARF_END_NAMESPACE
