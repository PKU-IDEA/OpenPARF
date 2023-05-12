/**
 * File              : ism_detailed_placer.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.14.2020
 * Last Modified Date: 07.14.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DETAILED_PLACER_HPP_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DETAILED_PLACER_HPP_

// C++ system library headers
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

// project headers
#include "database/placedb.h"
#include "ops/pin_utilization/src/pin_utilization_map_kernel.hpp"
#include "ops/rudy/src/rudy_kernel.hpp"

// local headers
#include "ops/ism_dp/src/ism_dp_db.hpp"
#include "ops/ism_dp/src/ism_dp_param.h"
#include "ops/ism_dp/src/ism_dp_type_trait.h"
#include "ops/ism_dp/src/ism_solver.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {
struct ISMDetailedPlaceState {
  std::vector<database::PlaceDB::CoordinateType> pos;             ///< #insts * 3, location of instances
  std::vector<database::PlaceDB::CoordinateType> pos_xy;          ///< #insts * 2, locations of instances
  std::vector<database::PlaceDB::CoordinateType> pin_pos;         ///< #pins * 2, pin locations
  std::vector<database::PlaceDB::CoordinateType> rudy_maps[2];    ///< bin_map_dim * 2, horizontal and vertical RUDY map
  std::vector<database::PlaceDB::CoordinateType> pin_util_map;    ///< bin_map_dim, pin utilization map
  std::vector<database::PlaceDB::CoordinateType> inst_num_pins;   ///< #insts, number of pins per inst
  std::vector<database::PlaceDB::CoordinateType>
          stretch_inst_sizes;   ///< #insts * 2, stretched inst sizes for pin utilization
  // std::vector<std::vector<uint8_t>> categoried_resources; ///< resources grouped in category
  std::vector<database::PlaceDB::IndexType>
          resource2site_types;   ///< map LUTL/LUTM/FF/Carry to 0, SSSIR types as 1, 2, ..., default is infinity
  Vector2D<database::PlaceDB::IndexType> valid_site_map;   ///< a complimentary data structure for SiteMap;
  ///< when [x, y] does not correspond to a valid site, this data structure provides the real site covering [x, y]
};

template<typename T>
class ISMDetailedPlacer {
 private:
  using IndexVector = std::vector<IndexType>;
  using IndexMap    = boost::container::flat_map<IndexType, IndexType>;

  /// Class to represent a CLB
  /// For pin access optimization and BLE alignment optimization
  struct CLB {
    struct BLE {
      explicit BLE(IndexType BLE_capacity) {
        lut.assign(BLE_capacity, kIndexTypeMax);
        ff.assign(BLE_capacity, kIndexTypeMax);
      }

      std::vector<IndexType>   lut;
      std::vector<IndexType>   ff;
      IndexType                ck      = kIndexTypeMax;
      IndexType                sr      = kIndexTypeMax;
      std::array<IndexType, 2> ce      = {{kIndexTypeMax, kIndexTypeMax}};
      IndexType                origIdx = kIndexTypeMax;
      IndexType                numPins = 0;
    };

    void getCtrlSet(std::array<IndexType, 2> &ck, std::array<IndexType, 2> &sr, std::array<IndexType, 4> &ce) const {
      std::fill(ck.begin(), ck.end(), kIndexTypeMax);
      std::fill(sr.begin(), sr.end(), kIndexTypeMax);
      std::fill(ce.begin(), ce.end(), kIndexTypeMax);
      for (IndexType i = 0; i < db_.numBLEsPerCLB(); ++i) {
        const auto &b  = ble[i];
        IndexType   lh = i / db_.numBLEsPerHalfCLB();
        ck[lh]         = (ck[lh] == kIndexTypeMax ? b.ck : ck[lh]);
        sr[lh]         = (sr[lh] == kIndexTypeMax ? b.sr : sr[lh]);
        for (IndexType k : {0, 1}) {
          IndexType idx = 2 * lh + k;
          ce[idx]       = (ce[idx] == kIndexTypeMax ? b.ce[k] : ce[idx]);
        }
      }
    }

    explicit CLB(database::PlaceDB const &db) : db_(db), ble(db.numBLEsPerCLB(), BLE(db.BLECapacity())) {}

    database::PlaceDB const &db_;
    std::vector<BLE>         ble;
  };

 public:
  /// Constructor
  ISMDetailedPlacer(database::PlaceDB const &                         db,
                    // ISMDetailedPlaceDB<T> db,
                    ISMDetailedPlaceParam                             param,
                    std::function<bool(uint32_t, uint32_t, uint32_t)> isCKAllowedInSite,
                    bool                                              honorClockConstraint,
                    int32_t                                           num_threads)
      : db_(db),
        param_(param),
        isCKAllowedInSite_(isCKAllowedInSite),
        honorClockConstraint_(honorClockConstraint),
        num_threads_(num_threads) {
    state_.pos.resize(db_.numInsts() * 3);
    state_.pin_pos.resize(db_.numPins() * 2);
    state_.pos_xy.resize(db_.numInsts() * 2);
    state_.rudy_maps[0].resize(param_.routeUtilBinDimX * param_.routeUtilBinDimY);
    state_.rudy_maps[1].resize(param_.routeUtilBinDimX * param_.routeUtilBinDimY);
    state_.pin_util_map.resize(param_.pinUtilBinDimX * param_.pinUtilBinDimY);
    state_.inst_num_pins.resize(db_.numInsts());
    state_.stretch_inst_sizes.resize(db_.numInsts() * 2);

    // Initialize data for pin utilization estimation
    for (IndexType i = 0; i < db_.numInsts(); ++i) {
      // Compute the pin weight
      // Since control set pins (e.g., CK/SR/CE) in FFs can be largely shared,
      // it is almost always an overestimation of using FF's pin count directly
      // To compensate this effect, we give FFs a pre-defined pin weight
      // For non-FF instances, we just use pin counts as the weights
      if (db_.isInstFF(i)) {
        state_.inst_num_pins[i] = param_.ffPinWeight;
      } else {
        state_.inst_num_pins[i] = db_.instPins().size2(i);
      }
    }

    // state_.categoried_resources.resize(toInteger(ResourceCategory::kUnknown));
    // for (uint32_t resource = 0; resource < db_.numResources(); ++resource) {
    //  state_.categoried_resources[db_.resourceCategory(resource)].push_back(resource);
    //}

    state_.resource2site_types.assign(db_.numResources(), kIndexTypeMax);
    IndexType max_site_type = 0;
    for (IndexType resource = 0; resource < db_.numResources(); ++resource) {
      auto category = db_.resourceCategory(resource);
      switch (category) {
        case ResourceCategory::kLUTL:
        case ResourceCategory::kLUTM:
        case ResourceCategory::kFF:
        case ResourceCategory::kCarry:
          state_.resource2site_types[resource] = 0;
          break;
        case ResourceCategory::kSSSIR:
          max_site_type += 1;
          state_.resource2site_types[resource] = max_site_type;
          break;
        default:
          break;
      }
    }

    auto const &layout   = db_.db()->layout();
    auto const &site_map = layout.siteMap();
    state_.valid_site_map.resize(site_map.width(), site_map.height(), kIndexTypeMax);
    for (auto const &site : site_map) {
      auto const &bbox = site.bbox();
      for (IndexType ix = bbox.xl(); ix < bbox.xh(); ++ix) {
        for (IndexType iy = bbox.yl(); iy < bbox.yh(); ++iy) {
          state_.valid_site_map(ix, iy) = site_map.index1D(site.siteMapId().x(), site.siteMapId().y());
        }
      }
    }
  }

  /// Top function to run DP
  void run(T *pos) {
    std::copy(pos, pos + db_.numInsts() * 3, state_.pos.data());

    if (param_.verbose > 0) {
      openparfPrint(kInfo, "--------------- CLB ISM ---------------\n");
    }
    updateRoutePinUtil();
    runCLBISM();
    if (param_.verbose > 0) {
      openparfPrint(kInfo, "--------------- BLE ISM ---------------\n");
    }
    updateRoutePinUtil();
    runBLEISM();

    if (param_.verbose > 0) {
      openparfPrint(kInfo, "----------- LUT/FF Pair ISM -----------\n");
    }
    updateRoutePinUtil();
    runPairISM();

    // Perform final intra-CLB optimization
    optimizeCLBs();

    // Check placement feasibility
    // db_.checkPlaceFeasibility();

    // copy back to pos
    std::copy(state_.pos.begin(), state_.pos.end(), pos);
  }

 protected:
  /// @brief a function for debug
  std::string getInstName(IndexType inst_id) const {
    auto const &design          = db_.db()->design();
    auto        top_module_inst = design.topModuleInst();
    openparfAssert(top_module_inst);
    // assume flat netlist for now
    auto const &netlist = top_module_inst->netlist();
    auto const &inst    = netlist.inst(db_.oldInstId(inst_id));
    return inst.attr().name();
  }
  IndexType siteResourceCapacity(IndexType site_id, IndexType resource) const {
    auto const &layout = db_.db()->layout();
    auto const &site   = layout.siteMap().at(site_id);
    openparfAssert(site);
    auto const &site_type = layout.siteType(*site);
    return site_type.resourceCapacity(resource);
  }
  std::vector<std::pair<IndexType, IndexType>> siteResourceCategoryCapacity(database::Site const &site,
                                                                            ResourceCategory      category) const {
    auto const &                                 layout    = db_.db()->layout();
    auto const &                                 site_type = layout.siteType(site);
    std::vector<std::pair<IndexType, IndexType>> resource_and_capacities;
    for (IndexType resource = 0; resource < db_.numResources(); ++resource) {
      auto cap = site_type.resourceCapacity(resource);
      if (cap) {
        auto cat = db_.resourceCategory(resource);
        if (cat == category) {
          resource_and_capacities.emplace_back(resource, cap);
        }
      }
    }
    return resource_and_capacities;
  }

  /// For a given node, get it's corresponding site Y coordiante
  // inline IndexType getNodeSiteY(IndexType inst_id, T inst_y) const {
  //  T height = getInstSize(inst_id).y();
  //  return IndexType(std::floor(inst_y / height) * height);
  //}

  /// For a given site, get its height
  /// The height for IO is set to 1.
  // RealType getSiteHeight(database::Site const& site) const {
  //  auto const& layout = db_.db()->layout();
  //  auto const& site_type = layout.siteType(site);
  //  for (IndexType resource = 0; resource < db_.numResources(); ++resource) {
  //    auto cap = site_type.resourceCapacity(resource);
  //    if (cap) {
  //      auto cat = db_.resourceCategory(resource);
  //      switch (cat) {
  //        case ResourceCategory::kLUTL:
  //        case ResourceCategory::kLUTM:
  //        case ResourceCategory::kFF:
  //        case ResourceCategory::kSSSIR:
  //          // debug, hard code
  //          if (resource == 3) {
  //            return 2.5;
  //          }
  //          return site.bbox().height();
  //        case ResourceCategory::kSSMIR:
  //          return 1.0;
  //        default:
  //          break;
  //      }
  //    }
  //  }
  //  return 1.0;
  //}

  database::Site const &getValidSite(IndexType ix, IndexType iy) const {
    auto        id1d   = state_.valid_site_map(ix, iy);
    auto const &layout = db_.db()->layout();
    auto const &site   = layout.siteMap().at(id1d);
    openparfAssert(site);
    return *site;
  }

  /// get height of an instance
  inline database::PlaceDB::SizeType getInstSize(IndexType inst_id) const {
    database::PlaceDB::CoordinateType w = 0;
    database::PlaceDB::CoordinateType h = 0;
    for (IndexType at = 0; at < db_.numAreaTypes(); ++at) {
      auto const &size = db_.instSize(inst_id, at);
      w                = std::max(w, size.x());
      h                = std::max(h, size.y());
    }
    return {w, h};
  }

  /// Update routing and pin utiizations
  void updateRoutePinUtil() {
    // Update routing utilization
    for (uint32_t i = 0; i < db_.numPins(); ++i) {
      int32_t inst_id           = db_.pin2Inst(i);
      state_.pin_pos[i * 2]     = state_.pos[inst_id * 3];
      state_.pin_pos[i * 2 + 1] = state_.pos[inst_id * 3 + 1];
    }
    rudyLauncher<database::PlaceDB::CoordinateType,
            typename std::remove_pointer<decltype(db_.netPins().data().data())>::type>(state_.pin_pos.data(),
            db_.netPins().indexBeginData().data(), db_.netPins().data().data(), db_.netWeights().data(),
            param_.routeUtilBinW, param_.routeUtilBinH, db_.diearea().xl(), db_.diearea().yl(), db_.diearea().xh(),
            db_.diearea().yh(), param_.routeUtilBinDimX, param_.routeUtilBinDimY, db_.numNets(),
            /*deterministic_flag=*/1, num_threads_, state_.rudy_maps[0].data(), state_.rudy_maps[1].data());
    // normalize to unit
    auto   bin_area              = param_.routeUtilBinW * param_.routeUtilBinH;
    double route_scale_factor[2] = {1.0 / (bin_area * param_.unitHoriRouteCap),
                                    1.0 / (bin_area * param_.unitVertRouteCap)};
    for (int32_t i = 0; i < 2; ++i) {
      for (auto &value : state_.rudy_maps[i]) {
        value *= route_scale_factor[i];
      }
    }

    // Update pin utilization
    RealType pinW = param_.pinUtilBinW * param_.pinStretchRatio;
    RealType pinH = param_.pinUtilBinH * param_.pinStretchRatio;
    for (uint32_t i = 0; i < db_.numInsts(); ++i) {
      // Compute the pin bounding box
      // To make the pin density map smooth, we stretch each pin to a ratio of the pin utilization bin
      auto     node_x     = state_.pos[i * 3];
      auto     node_y     = state_.pos[i * 3 + 1];
      RealType w          = pinW;
      RealType h          = pinH;
      auto     categories = db_.instResourceCategory(i);
      if (std::find(categories.begin(), categories.end(), ResourceCategory::kSSSIR) != categories.end()) {
        h = std::max(h, (RealType) getInstSize(i).y());
      }
      state_.pos_xy[i * 2]                 = node_x;
      state_.pos_xy[i * 2 + 1]             = node_y;
      state_.stretch_inst_sizes[i * 2]     = w;
      state_.stretch_inst_sizes[i * 2 + 1] = h;
    }
    pinDemandMapLauncher<database::PlaceDB::CoordinateType>(state_.pos_xy.data(), state_.stretch_inst_sizes.data(),
            state_.inst_num_pins.data(), db_.diearea().xl(), db_.diearea().yl(), db_.diearea().xh(), db_.diearea().yh(),
            param_.pinUtilBinW, param_.pinUtilBinH, param_.pinUtilBinDimX, param_.pinUtilBinDimY,
            std::make_pair(0, db_.numInsts()),
            /*deterministic_clag=*/1, num_threads_, state_.pin_util_map.data());
    // normalize to unit
    auto pin_scale_factor = 1.0 / (bin_area * param_.unitPinCap);
    for (auto &value : state_.pin_util_map) {
      value *= pin_scale_factor;
    }
  }

  /// Top function to run CLB ISM
  void runCLBISM() {
    // Initialize the ISM parameters
    ISMParam param;
    initCLBISMParam(param);

    // Initialize the ISM problem
    std::vector<IndexVector> clusters;
    ISMProblem               prob;
    initCLBISMProblem(clusters, prob, honorClockConstraint_);

    // Perform the ISM
    ISMSolver solver(prob, param, num_threads_, honorClockConstraint_);
    solver.buildISMProblem(prob);
    solver.run();

    // Write the CLB ISM solution back to the netlist
    auto zGetter = [&](IndexType node_id, const ISMSolver::ISMInstanceSol &sol) { return state_.pos[node_id * 3 + 2]; };
    writeISMSolution(zGetter, clusters, prob, solver);
  }

  /// Initialize the CLB ISM problem
  void initCLBISMProblem(std::vector<IndexVector> &clusters, ISMProblem &prob, bool honorClockConstraints) {
    // Initialize netlist
    IndexVector         nodeToCluster;
    Vector2D<IndexType> siteToCluster(db_.siteMapDim().x(), db_.siteMapDim().y(), kIndexTypeMax);
    auto                siteIdGetter = [&](IndexType node_id, T node_x, T node_y, T node_z) {
      openparfAssert(node_id < db_.numInsts());
      auto categories = db_.instResourceCategory(node_id);
      if (std::find(categories.begin(), categories.end(), ResourceCategory::kSSSIR) == categories.end()) {
        return siteToCluster.xyToIndex(node_x, node_y);
      }
      auto const &site = getValidSite(node_x, node_y);
      return siteToCluster.xyToIndex(site.siteMapId().x(), site.siteMapId().y());
    };
    initISMProblemNetlist(siteIdGetter, clusters, nodeToCluster, siteToCluster, prob, honorClockConstraint_);

    // Initialize insatnce-to-site assignment, instance types, and site locations
    auto siteTypeGetter = [&](IndexType site_id) -> IndexType {
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(site_id);
      if (site) {
        auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
        if (!lutm_cap.empty()) {
          return 1;
        }
        auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
        if (!lutl_cap.empty()) {
          return 0;
        }
        auto sssir_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kSSSIR);
        if (!sssir_cap.empty()) {
          return 1 + state_.resource2site_types[sssir_cap[0].first];
        }
      }
      return kIndexTypeMax;
    };
    auto addFiller = [&](IndexType site_id) {
      auto loc = siteToCluster.indexToXY(site_id);
      return siteTypeGetter(site_id) != kIndexTypeMax &&
             std::max(state_.rudy_maps[0][loc.x() * param_.routeUtilBinDimY + loc.y()],
                      state_.rudy_maps[1][loc.x() * param_.routeUtilBinDimY + loc.y()]) <
                     param_.routeUtilToFixWhiteSpace &&
             state_.pin_util_map[loc.x() * param_.pinUtilBinDimY + loc.y()] < param_.pinUtilToFixWhiteSpace;
    };
    auto siteXYGetter = [&](IndexType site_id) {
      auto        loc  = siteToCluster.indexToXY(site_id);
      auto const &site = getValidSite(loc.x(), loc.y());
      auto const &bbox = site.bbox();
      return XY<RealType>((bbox.xl() + bbox.xh()) * 0.5, (bbox.yl() + bbox.yh()) * 0.5);
    };
    initISMProblemSiteMap(siteTypeGetter, addFiller, siteXYGetter, clusters, siteToCluster, prob);
    initISMProblemFixedInst(clusters, prob);

    // Initialize pin offsets
    initISMProblemPinOffset(prob);

    if (honorClockConstraint_) {
      prob.isCKAllowedInSite = isCKAllowedInSite_;
    }
    // Set other unused information for CLB ISM
    IndexType numInsts = prob.instTypes.size();
    prob.instCKSR.assign(numInsts, kIndexTypeMax);
    prob.instCE.assign(numInsts, ISMProblem::CESet{kIndexTypeMax, kIndexTypeMax});
    prob.instToMates.assign(numInsts, IndexVector());
    prob.siteToCtrlSet.clear();
    prob.siteToCtrlSet.resize(prob.siteXYs.sizes(), kIndexTypeMax);
    prob.siteToMate.clear();
    prob.siteToMate.resize(prob.siteXYs.sizes(), kIndexTypeMax);
  }

  /// Initialize CLB ISM parameters
  void initCLBISMParam(ISMParam &param) {
    // Message printing
    param.verbose           = 1;

    // These paramters should not be changed for CLB ISM
    param.numSitePerCtrlSet = kIndexTypeMax;
    param.siteXStep         = 1;
    param.siteYInterval     = 1;
    param.xWirelenWt        = param_.xWirelenWt;
    param.yWirelenWt        = param_.yWirelenWt;

    // Tunable parameters
    param.maxNetDegree      = 16;
    param.maxRadius         = 10;
    param.maxIndepSetSize   = 100;
    param.mateCredit        = 0.0;
  }

  /// Top function to run BLE ISM
  void runBLEISM() {
    // Initialize the ISM parameters
    ISMParam param;
    initBLEISMParam(param);
    // Initialize the ISM problem
    std::vector<IndexVector> clusters;
    ISMProblem               prob;
    initBLEISMProblem(clusters, prob);

    // Perform the ISM
    ISMSolver solver(prob, param, num_threads_, honorClockConstraint_);
    solver.buildISMProblem(prob);
    solver.run();

    // Write the BLE ISM solution back to the netlist
    auto zGetter = [&](IndexType node_id, const ISMSolver::ISMInstanceSol &sol) -> IndexType {
      IndexType node_z     = state_.pos[node_id * 3 + 2];
      auto      categories = db_.instResourceCategory(node_id);
      for (auto category : categories) {
        switch (category) {
          case ResourceCategory::kLUTM:
            return node_z;
          case ResourceCategory::kLUTL: {
            openparfAssert(!sol.isFixed());
            return (prob.siteXYs.indexToY(sol.siteId) % db_.numBLEsPerCLB()) * db_.BLECapacity() +
                   node_z % db_.BLECapacity();
          }
          case ResourceCategory::kFF: {
            openparfAssert(!sol.isFixed());
            IndexType z = (prob.siteXYs.indexToY(sol.siteId) % db_.numBLEsPerCLB()) * db_.BLECapacity() +
                          node_z % db_.BLECapacity();
            if (sol.ceFlip) {
              z += (node_z % 2 ? -1 : 1);
            }
            return z;
          }
          default:
            break;
        }
      }
      return node_z;
    };
    writeISMSolution(zGetter, clusters, prob, solver);
  }

  /// Initialize the BLE ISM problem
  void initBLEISMProblem(std::vector<IndexVector> &clusters, ISMProblem &prob) {
    // Initialize netlist
    IndexVector         nodeToCluster;
    Vector2D<IndexType> siteToCluster(db_.siteMapDim().x(), db_.siteMapDim().y() * db_.numBLEsPerCLB(), kIndexTypeMax);
    auto                siteIdGetter = [&](IndexType node_id, T node_x, T node_y, T node_z) {
      auto const &site       = getValidSite(node_x, node_y);
      auto        categories = db_.instResourceCategory(node_id);
      for (auto category : categories) {
        switch (category) {
          case ResourceCategory::kLUTL:
          case ResourceCategory::kFF: {
            return siteToCluster.xyToIndex(site.siteMapId().x(),
                                           site.siteMapId().y() * db_.numBLEsPerCLB() + node_z / db_.BLECapacity());
          }
          case ResourceCategory::kSSSIR: {
            return siteToCluster.xyToIndex(site.siteMapId().x(), site.siteMapId().y() * db_.numBLEsPerCLB());
          }
          default:
            break;
        }
      }
      return siteToCluster.xyToIndex(site.siteMapId().x(), site.siteMapId().y() * db_.numBLEsPerCLB());
    };
    initISMProblemNetlist(siteIdGetter, clusters, nodeToCluster, siteToCluster, prob, honorClockConstraint_);

    // Initialize insatnce-to-site assignment, instance types, and site locations
    auto siteTypeGetter = [&](IndexType site_id) -> IndexType {
      auto        loc    = siteToCluster.indexToXY(site_id);
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(loc.x(), loc.y() / db_.numBLEsPerCLB());
      if (site) {
        auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
        if (!lutm_cap.empty()) {
          return kIndexTypeMax;
        }
        auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
        if (!lutl_cap.empty()) {
          return 0;
        }
        auto sssir_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kSSSIR);
        if (!sssir_cap.empty()) {
          return 1 + state_.resource2site_types[sssir_cap[0].first];
        }
      }
      return kIndexTypeMax;
    };
    auto addFiller = [&](IndexType site_id) {
      IndexType   binX   = siteToCluster.indexToX(site_id);
      IndexType   binY   = siteToCluster.indexToY(site_id);
      IndexType   siteX  = binX;
      IndexType   siteY  = binY / db_.numBLEsPerCLB();
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(siteX, siteY);
      if (site) {
        if (!siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL).empty() ||
            !siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM).empty()) {
          return std::max(state_.rudy_maps[0][siteX * param_.routeUtilBinDimY + siteY],
                          state_.rudy_maps[1][siteX * param_.routeUtilBinDimY + siteY]) <
                         param_.routeUtilToFixWhiteSpace &&
                 state_.pin_util_map[siteX * param_.pinUtilBinDimY + siteY] < param_.pinUtilToFixWhiteSpace;
        } else if (!siteResourceCategoryCapacity(site.value(), ResourceCategory::kSSSIR).empty()) {
          return (binY % db_.numBLEsPerCLB()) == 0;
        }
      }
      return false;
    };
    auto siteXYGetter = [&](IndexType site_id) {
      auto        loc  = siteToCluster.indexToXY(site_id);
      auto const &site = getValidSite(loc.x(), loc.y() / db_.numBLEsPerCLB());
      auto const &bbox = site.bbox();
      return XY<RealType>((bbox.xl() + bbox.xh()) * 0.5, (bbox.yl() + bbox.yh()) * 0.5);
      // auto const& layout = db_.db()->layout();
      // auto const& site = layout.siteMap().at(loc.x(), loc.y() / db_.numBLEsPerCLB());
      // openparfAssert(site);
      // if (!siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL).empty()
      //    || !siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM).empty()) {
      //  return XY<RealType>(loc.x() + 0.5, (loc.y() + 0.5) / db_.numBLEsPerCLB());
      //}
      // RealType siteH = getSiteHeight(site.value());
      // return XY<RealType>(loc.x() + 0.5, (std::round(loc.y() / db_.numBLEsPerCLB() / siteH) + 0.5) * siteH);
    };
    initISMProblemSiteMap(siteTypeGetter, addFiller, siteXYGetter, clusters, siteToCluster, prob);
    initISMProblemFixedInst(clusters, prob);

    // Initialize BLE control sets
    auto ctrlSetGetter = [&](IndexType site_id) -> IndexType {
      auto        loc    = siteToCluster.indexToXY(site_id);
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(loc.x(), loc.y() / db_.numBLEsPerCLB());
      if (site) {
        if (!siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL).empty() ||
            !siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM).empty()) {
          IndexType ctrlSetCap = db_.numBLEsPerCLB() / 2;
          return loc.x() * siteToCluster.ySize() / ctrlSetCap + loc.y() / ctrlSetCap;
        }
      }
      return kIndexTypeMax;
    };
    initISMProblemCtrlSets(ctrlSetGetter, clusters, prob);
    if (honorClockConstraint_) prob.isCKAllowedInSite = isCKAllowedInSite_;

    // Initialize pin offsets
    initISMProblemPinOffset(prob);

    // Set other unused information for BLE ISM
    prob.instToMates.assign(prob.instTypes.size(), IndexVector());
    prob.siteToMate.clear();
    prob.siteToMate.resize(prob.siteXYs.sizes(), kIndexTypeMax);
  }

  /// Initialize BLE ISM parameters
  void initBLEISMParam(ISMParam &param) {
    // Message printing
    param.verbose           = 1;

    // These paramters should not be changed for BLE ISM
    param.numSitePerCtrlSet = db_.numBLEsPerCLB() / db_.numControlSetsPerCLB();
    param.siteXStep         = 1;
    param.siteYInterval     = db_.numBLEsPerCLB();
    param.xWirelenWt        = param_.xWirelenWt;
    param.yWirelenWt        = param_.yWirelenWt;

    // Tunable parameters
    param.maxNetDegree      = 24;
    param.maxRadius         = 7;
    param.maxIndepSetSize   = 50;
    param.mateCredit        = 0.0;
  }

  /// Top function to run LUT/FF pair ISM
  void runPairISM() {
    // Initialize the ISM parameters
    ISMParam param;
    initPairISMParam(param);

    // Initialize the ISM problem
    std::vector<IndexVector> clusters;
    ISMProblem               prob;
    initPairISMProblem(param, clusters, prob);

    // Perform the ISM
    ISMSolver solver(prob, param, num_threads_, honorClockConstraint_);
    solver.buildISMProblem(prob);
    solver.run();

    // Write the LUT/FF pair ISM solution back to the netlist
    auto zGetter = [&](IndexType node_id, const ISMSolver::ISMInstanceSol &sol) -> IndexType {
      IndexType node_z     = state_.pos[node_id * 3 + 2];
      auto      categories = db_.instResourceCategory(node_id);
      for (auto category : categories) {
        switch (category) {
          case ResourceCategory::kLUTM:
            return node_z;
          case ResourceCategory::kLUTL: {
            openparfAssert(!sol.isFixed());
            return (prob.siteXYs.indexToY(sol.siteId) % db_.numBLEsPerCLB()) * db_.BLECapacity() +
                   node_z % db_.BLECapacity();
          }
          case ResourceCategory::kFF: {
            openparfAssert(!sol.isFixed());
            IndexType z = (prob.siteXYs.indexToY(sol.siteId) % db_.numBLEsPerCLB()) * db_.BLECapacity() +
                          node_z % db_.BLECapacity();
            if (sol.ceFlip) {
              z += (node_z % 2 ? -1 : 1);
            }
            return z;
          }
          default:
            break;
        }
      }
      return node_z;
    };
    writeISMSolution(zGetter, clusters, prob, solver);
  }

  /// Initialize the LUT/FF pair ISM problem
  void initPairISMProblem(const ISMParam &param, std::vector<IndexVector> &clusters, ISMProblem &prob) {
    // Initialize netlist
    IndexVector         nodeToCluster;
    Vector2D<IndexType> siteToCluster(db_.siteMapDim().x() * 2,
                                      db_.siteMapDim().y() * db_.numBLEsPerCLB(),
                                      kIndexTypeMax);
    auto                siteIdGetter = [&](IndexType inst_id, T inst_x, T inst_y, T inst_z) {
      auto const &site       = getValidSite(inst_x, inst_y);
      auto        categories = db_.instResourceCategory(inst_id);
      for (auto category : categories) {
        switch (category) {
          case ResourceCategory::kLUTL:
          case ResourceCategory::kLUTM:
            return siteToCluster.xyToIndex(site.siteMapId().x() * 2,
                                           site.siteMapId().y() * db_.numBLEsPerCLB() + inst_z / db_.BLECapacity());
            // return siteToCluster.xyToIndex((IndexType)inst_x * 2,
            //    (IndexType)inst_y * db_.numBLEsPerCLB() + inst_z / db_.BLECapacity());
          case ResourceCategory::kFF:
            return siteToCluster.xyToIndex(site.siteMapId().x() * 2 + 1,
                                           site.siteMapId().y() * db_.numBLEsPerCLB() + inst_z / db_.BLECapacity());
            // return siteToCluster.xyToIndex((IndexType)inst_x * 2 + 1,
            //    (IndexType)inst_y * db_.numBLEsPerCLB() + inst_z / db_.BLECapacity());
          case ResourceCategory::kSSSIR:
            return siteToCluster.xyToIndex(site.siteMapId().x() * 2, site.siteMapId().y() * db_.numBLEsPerCLB());
            // return siteToCluster.xyToIndex((IndexType)inst_x * 2, getNodeSiteY(inst_id, inst_y) *
            // db_.numBLEsPerCLB());
          default:
            break;
        }
      }
      return siteToCluster.xyToIndex(site.siteMapId().x() * 2, site.siteMapId().y() * db_.numBLEsPerCLB());
    };
    initISMProblemNetlist(siteIdGetter, clusters, nodeToCluster, siteToCluster, prob, honorClockConstraint_);

    // Initialize insatnce-to-site assignment, instance types, and site locations
    auto siteTypeGetter = [&](IndexType site_id) -> IndexType {
      auto        loc    = siteToCluster.indexToXY(site_id);
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(loc.x() / 2, loc.y() / db_.numBLEsPerCLB());
      if (site) {
        auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
        if (!lutm_cap.empty()) {
          return kIndexTypeMax;
        }
        auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
        if (!lutl_cap.empty()) {
          return loc.x() % 2;
        }
        auto sssir_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kSSSIR);
        if (!sssir_cap.empty()) {
          return 3 + state_.resource2site_types[sssir_cap[0].first];
        }
      }
      return kIndexTypeMax;
      // if (db_.site_LUT_capacities[site_id]) {
      //  openparfAssert(db_.site_FF_capacities[site_id]);
      //  return loc.x() % 2;
      //} else if (db_.site_DSP_capacities[site_id]) {
      //  return 2;
      //} else if (db_.site_RAM_capacities[site_id]) {
      //  return 3;
      //} else {
      //  return kIndexTypeMax;
      //}
    };
    auto addFiller = [&](IndexType site_id) {
      IndexType   binX   = siteToCluster.indexToX(site_id);
      IndexType   binY   = siteToCluster.indexToY(site_id);
      IndexType   siteX  = binX / 2;
      IndexType   siteY  = binY / db_.numBLEsPerCLB();
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(siteX, siteY);
      if (site) {
        auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
        auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
        if (!lutl_cap.empty() || !lutm_cap.empty()) {
          return std::max(state_.rudy_maps[0][siteX * param_.routeUtilBinDimY + siteY],
                          state_.rudy_maps[1][siteX * param_.routeUtilBinDimY + siteY]) <
                         param_.routeUtilToFixWhiteSpace &&
                 state_.pin_util_map[siteX * param_.pinUtilBinDimY + siteY] < param_.pinUtilToFixWhiteSpace;
        }
        auto sssir_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kSSSIR);
        if (!sssir_cap.empty()) {
          return (binX % 2) == 0 && (binY % db_.numBLEsPerCLB()) == 0;
        }
      }
      return false;
    };
    auto siteXYGetter = [&](IndexType site_id) {
      auto        loc  = siteToCluster.indexToXY(site_id);
      auto const &site = getValidSite(loc.x() / 2, loc.y() / db_.numBLEsPerCLB());
      auto const &bbox = site.bbox();
      return XY<RealType>((bbox.xl() + bbox.xh()) * 0.5, (bbox.yl() + bbox.yh()) * 0.5);
      // auto const& layout = db_.db()->layout();
      // auto const& site = layout.siteMap().at(loc.x() / 2, loc.y() / db_.numBLEsPerCLB());
      // openparfAssert(site);
      // auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
      // auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
      // if (!lutl_cap.empty() || !lutm_cap.empty()) {
      //  return XY<RealType>(loc.x() / 2 + 0.5, (loc.y() + 0.5) / db_.numBLEsPerCLB());
      //}
      // RealType siteH = getSiteHeight(site.value());
      // return XY<RealType>(loc.x() / 2 + 0.5, (std::round(loc.y() / db_.numBLEsPerCLB() / siteH) + 0.5) * siteH);
    };
    initISMProblemSiteMap(siteTypeGetter, addFiller, siteXYGetter, clusters, siteToCluster, prob);
    initISMProblemFixedInst(clusters, prob);

    // Initialize BLE control sets
    auto ctrlSetGetter = [&](IndexType site_id) -> IndexType {
      auto        loc    = siteToCluster.indexToXY(site_id);
      auto const &layout = db_.db()->layout();
      auto const &site   = layout.siteMap().at(loc.x() / 2, loc.y() / db_.numBLEsPerCLB());
      if (site) {
        auto lutl_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTL);
        auto lutm_cap = siteResourceCategoryCapacity(site.value(), ResourceCategory::kLUTM);
        if ((loc.x() % 2) && (!lutl_cap.empty() || !lutm_cap.empty())) {
          IndexType ctrlSetCap = db_.numBLEsPerCLB() / 2;
          return loc.x() / 2 * siteToCluster.ySize() / ctrlSetCap + loc.y() / ctrlSetCap;
        }
      }
      return kIndexTypeMax;
    };
    initISMProblemCtrlSets(ctrlSetGetter, clusters, prob);

    // Initialize pin offsets
    initISMProblemPinOffset(prob);

    // Initialize instance/site mates
    initISMProblemMates(param, nodeToCluster, prob);
    if (honorClockConstraint_) prob.isCKAllowedInSite = isCKAllowedInSite_;
  }

  /// Initialize LUT/FF pair ISM parameters
  void initPairISMParam(ISMParam &param) {
    // Message printing
    param.verbose           = 1;

    // These paramters should not be changed for BLE ISM
    param.numSitePerCtrlSet = db_.numBLEsPerCLB() / db_.numControlSetsPerCLB();
    param.siteXStep         = 2;
    param.siteYInterval     = db_.numBLEsPerCLB();
    param.xWirelenWt        = param_.xWirelenWt;
    param.yWirelenWt        = param_.yWirelenWt;

    // Tunable parameters
    param.maxNetDegree      = 32;
    param.maxRadius         = 7;
    param.maxIndepSetSize   = 50;
    param.mateCredit        = 2.0;
  }

  /// Initialize ISM problem netlist for a given clustering solution (e.g., CLB/BLE...)
  /// @param  siteIdGetter   a function to get node-to-site mapping
  /// @param  clusters       contains the resulting clusters after function call
  /// @param  siteToCluster  contains the site-to-cluster mapping after function call, it must be allocated before the
  /// function call
  /// @param  prob           the ISM problem to be constructed
  void initISMProblemNetlist(std::function<IndexType(IndexType, T, T, T)> siteIdGetter,
                             std::vector<IndexVector> &                   clusters,
                             IndexVector &                                nodeToCluster,
                             Vector2D<IndexType> &                        siteToCluster,
                             ISMProblem &                                 prob,
                             bool                                         honorClockConstraints = false) const {
    // Get instances in each cluster
    clusters.clear();
    prob.instCKs.clear();
    nodeToCluster.assign(db_.numInsts(), kIndexTypeMax);

    for (IndexType inst_id = 0; inst_id < db_.numInsts(); ++inst_id) {
      if (param_.fixedMask[inst_id]) {
        IndexType idx = clusters.size();
        clusters.emplace_back();
        if (honorClockConstraints) {
          prob.instCKs.emplace_back();
        }
        clusters[idx].push_back(inst_id);
        if (honorClockConstraints) {
          auto const &iter = db_.instToClocks()[inst_id];
          prob.instCKs[idx].insert(iter.begin(), iter.end());
        }
        nodeToCluster[inst_id] = idx;
      } else {
        auto  inst_x  = state_.pos[inst_id * 3];
        auto  inst_y  = state_.pos[inst_id * 3 + 1];
        auto  inst_z  = state_.pos[inst_id * 3 + 2];
        auto  site_id = siteIdGetter(inst_id, inst_x, inst_y, inst_z);
        auto &idx     = siteToCluster(site_id);
        if (idx == kIndexTypeMax) {
          idx = clusters.size();
          clusters.emplace_back();
          if (honorClockConstraints) prob.instCKs.emplace_back();
        }
        clusters[idx].push_back(inst_id);
        if (honorClockConstraints) {
          auto const &iter = db_.instToClocks()[inst_id];
          prob.instCKs[idx].insert(iter.begin(), iter.end());
        }
        nodeToCluster[inst_id] = idx;
      }
    }

    prob.pinToInst.resize(db_.numPins());
    prob.pinToNet.resize(db_.numPins());
    prob.netWts.resize(db_.numNets());
    for (IndexType pin_id = 0; pin_id < db_.numPins(); ++pin_id) {
      IndexType idx       = pin_id;
      IndexType node_id   = db_.pin2Inst(pin_id);
      prob.pinToInst[idx] = nodeToCluster[node_id];
      prob.pinToNet[idx]  = db_.pin2Net(pin_id);
    }
    for (IndexType net_id = 0; net_id < db_.numNets(); ++net_id) {
      prob.netWts[net_id] = db_.netWeight(net_id);
    }
  }

  /// Initialize insatnce-to-site assignment, instance types, and site locations
  /// @param  siteTypeGetter  a function to get site-to-site type mapping
  /// @param  addFiller       a function to decide whether to add filler as a given site
  /// @param  siteXYGetter    a function to get site locations
  /// @param  clusters        clusters for ISM
  /// @param  siteToCluster   contains the site-to-cluster mapping (including fillers) after function call
  /// @param  prob            the ISM problem to be constructed
  void initISMProblemSiteMap(std::function<IndexType(IndexType)>    siteTypeGetter,
                             std::function<bool(IndexType)>         addFiller,
                             std::function<XY<RealType>(IndexType)> siteXYGetter,
                             const std::vector<IndexVector> &       clusters,
                             Vector2D<IndexType> &                  siteToCluster,
                             ISMProblem &                           prob) const {
    IndexType numInsts = clusters.size();

    prob.siteXYs.resize(siteToCluster.sizes());
    for (IndexType site_id = 0; site_id < siteToCluster.size(); ++site_id) {
      prob.siteXYs[site_id] = siteXYGetter(site_id);
    }

    prob.instTypes.resize(numInsts + siteToCluster.size(), kIndexTypeMax);
    prob.instToSite.resize(numInsts + siteToCluster.size(), kIndexTypeMax);
    for (IndexType site_id = 0; site_id < siteToCluster.size(); ++site_id) {
      auto &id = siteToCluster[site_id];
      if (id == kIndexTypeMax && addFiller(site_id)) {
        // Add a filler instance at this site
        id = numInsts++;
      }
      if (id != kIndexTypeMax) {
        prob.instTypes[id]  = siteTypeGetter(site_id);
        prob.instToSite[id] = site_id;
      }
    }
    prob.instTypes.resize(numInsts);
    prob.instToSite.resize(numInsts);
  }

  void initISMProblemFixedInst(const std::vector<IndexVector> &clusters, ISMProblem &prob) const {
    IndexType numInsts = prob.instTypes.size();
    prob.instXYs.resize(numInsts);
    for (int32_t i = 0; i < numInsts; i++) {
      if (prob.instToSite[i] == kIndexTypeMax) {
        openparfAssert(i < clusters.size());
        openparfAssert(clusters[i].size() == 1);
        int32_t  node_id = clusters[i][0];
        RealType xx      = state_.pos[node_id * 3];
        RealType yy      = state_.pos[node_id * 3 + 1];
        prob.instXYs[i]  = XY<RealType>(xx, yy);
      }
    }
  }

  /// Initialize ISM problem instance/site control sets
  /// @param  ctrlSetGetter  a function to get site-to-control set mapping
  /// @param  clusters       clusters for ISM
  /// @param  prob           the ISM problem to be constructed
  void initISMProblemCtrlSets(std::function<IndexType(IndexType)> ctrlSetGetter,
                              const std::vector<IndexVector> &    clusters,
                              ISMProblem &                        prob) const {
    // Set instance control sets
    prob.instCKSR.assign(prob.instTypes.size(), kIndexTypeMax);
    prob.instCE.assign(prob.instTypes.size(), ISMProblem::CESet{kIndexTypeMax, kIndexTypeMax});

    boost::container::flat_map<IndexType, IndexMap> cksrMapping;
    IndexMap                                        ceMapping;
    IndexType                                       cksrId = 0;
    IndexType                                       ceId   = 0;
    for (IndexType i = 0; i < clusters.size(); ++i) {
      for (IndexType node_id : clusters[i]) {
        auto categories = db_.instResourceCategory(node_id);
        if (std::find(categories.begin(), categories.end(), ResourceCategory::kFF) == categories.end()) {
          continue;
        }

        // Get net index of CK/SR/CE
        IndexType ck = kIndexTypeMax;
        IndexType sr = kIndexTypeMax;
        IndexType ce = kIndexTypeMax;
        for (IndexType j = 0; j < db_.instPins().size2(node_id); ++j) {
          auto pin_id = db_.instPins().at(node_id, j);
          switch (db_.pinSignalType(pin_id)) {
            case SignalType::kClock:
              ck = db_.pin2Net(pin_id);
              break;
            case SignalType::kControlSR:
              sr = db_.pin2Net(pin_id);
              break;
            case SignalType::kControlCE:
              ce = db_.pin2Net(pin_id);
              break;
            default:
              break;
          }
        }
        // Set CK/SR
        auto ckIt = cksrMapping.find(ck);
        if (ckIt == cksrMapping.end()) {
          prob.instCKSR[i]    = cksrId;
          cksrMapping[ck][sr] = cksrId++;
        } else {
          auto &srMap = ckIt->second;
          auto  srIt  = srMap.find(sr);
          if (srIt == srMap.end()) {
            prob.instCKSR[i] = cksrId;
            srMap[sr]        = cksrId++;
          } else {
            prob.instCKSR[i] = srIt->second;
          }
        }

        // Set the CE mapping
        IndexType k    = IndexType(state_.pos[node_id * 3 + 2]) % 2;
        auto      ceIt = ceMapping.find(ce);
        if (ceIt == ceMapping.end()) {
          prob.instCE[i][k] = ceId;
          ceMapping[ce]     = ceId++;
        } else {
          prob.instCE[i][k] = ceIt->second;
        }
      }
    }

    // Set site control sets
    prob.siteToCtrlSet.resize(prob.siteXYs.sizes());
    for (IndexType siteId = 0; siteId < prob.siteToCtrlSet.size(); ++siteId) {
      prob.siteToCtrlSet[siteId] = ctrlSetGetter(siteId);
    }
  }

  /// Initialize pin offsets
  /// We set all pin offsets to 0, since we already put instances at the center of sites
  void initISMProblemPinOffset(ISMProblem &prob) const {
    prob.pinOffsets.assign(db_.numPins(), XY<RealType>(0.0, 0.0));
  }

  /// Initialize instance/site mates
  /// This function is only for LUT/FF pair ISM
  /// We want LUT/FF pairs that have LUT.output->FF.input nets be mates
  void initISMProblemMates(const ISMParam &param, const IndexVector &nodeToCluster, ISMProblem &prob) const {
    // Set instance mates
    prob.instToMates.assign(prob.instTypes.size(), IndexVector());
    for (IndexType node_id = 0; node_id < db_.numInsts(); ++node_id) {
      auto categories = db_.instResourceCategory(node_id);
      if (std::find(categories.begin(), categories.end(), ResourceCategory::kLUTL) == categories.end() &&
          std::find(categories.begin(), categories.end(), ResourceCategory::kLUTM) == categories.end()) {
        continue;
      }
      // find the output pin of the instance
      IndexType outPin = kIndexTypeMax;
      for (IndexType j = 0; j < db_.instPins().size2(node_id); ++j) {
        auto pin_id = db_.instPins().at(node_id, j);
        if (db_.pinSignalDirect(pin_id) == SignalDirection::kOutput) {
          outPin = pin_id;
          break;
        }
      }
      if (outPin == kIndexTypeMax) {
        // this LUT instance has no output pin
        continue;
      }
      // get the net corresponding to the output pin
      IndexType outNet     = db_.pin2Net(outPin);
      auto      net_degree = db_.netPins().size2(outNet);
      if (net_degree > param.maxNetDegree) {
        continue;
      }
      IndexType currId = nodeToCluster[node_id];
      for (IndexType j = 0; j < db_.netPins().size2(outNet); ++j) {
        auto net_pin_id  = db_.netPins().at(outNet, j);
        auto net_node_id = db_.pin2Inst(net_pin_id);
        if (db_.pinSignalDirect(net_pin_id) == SignalDirection::kInput) {
          auto categories = db_.instResourceCategory(net_node_id);
          if (std::find(categories.begin(), categories.end(), ResourceCategory::kFF) != categories.end()) {
            IndexType mateId = nodeToCluster[net_node_id];
            prob.instToMates[currId].push_back(mateId);
            prob.instToMates[mateId].push_back(currId);
          }
        }
      }
    }

    // Set site mates
    prob.siteToMate.resize(prob.siteXYs.sizes());
    for (IndexType i = 0; i < prob.siteToMate.xSize(); i += 2) {
      for (IndexType j = 0; j < prob.siteToMate.ySize(); ++j) {
        prob.siteToMate(i, j)     = prob.siteToMate.xyToIndex(i + 1, j);
        prob.siteToMate(i + 1, j) = prob.siteToMate.xyToIndex(i, j);
      }
    }
  }

  // Write the ISM solution back to the netlist
  /// @param  zGetter   a function to get nodes' z values
  /// @param  clusters  clusters for ISM
  /// @param  prob      the ISM problem
  /// @param  solver    the ISM solver
  void writeISMSolution(std::function<IndexType(IndexType, const ISMSolver::ISMInstanceSol &)> zGetter,
                        const std::vector<IndexVector> &                                       clusters,
                        const ISMProblem &                                                     prob,
                        const ISMSolver &                                                      solver) {
    for (IndexType i = 0; i < clusters.size(); ++i) {
      const auto &sol = solver.instSol(i);
      if (sol.isFixed()) {
        continue;
      }
      const auto &loc = prob.siteXYs[sol.siteId];
      for (IndexType node_id : clusters[i]) {
        state_.pos[node_id * 3]     = loc.x();
        state_.pos[node_id * 3 + 1] = loc.y();
        state_.pos[node_id * 3 + 2] = zGetter(node_id, sol);
      }
    }
  }

  void optimizeCLBs() {
    // Collect LUT/FF in each CLB
    // Collect control sets of each BLE
    // Collect pin counts of each BLE but ignore control pins (e.g., CK/SR/CE)
    Vector2D<CLB> clbMap(db_.siteMapDim().x(), db_.siteMapDim().y(), CLB(db_));
    for (IndexType inst_id = 0; inst_id < db_.numInsts(); ++inst_id) {
      auto inst_x     = state_.pos[inst_id * 3];
      auto inst_y     = state_.pos[inst_id * 3 + 1];
      auto inst_z     = IndexType(state_.pos[inst_id * 3 + 2]);
      auto categories = db_.instResourceCategory(inst_id);
      if (std::find(categories.begin(), categories.end(), ResourceCategory::kLUTL) != categories.end() ||
          std::find(categories.begin(), categories.end(), ResourceCategory::kLUTM) != categories.end()) {
        auto &ble                           = clbMap(inst_x, inst_y).ble[inst_z / db_.BLECapacity()];
        ble.lut[inst_z % db_.BLECapacity()] = inst_id;
        ble.numPins += db_.instPins().size2(inst_id);
      } else if (std::find(categories.begin(), categories.end(), ResourceCategory::kFF) != categories.end()) {
        auto &ble                          = clbMap(inst_x, inst_y).ble[inst_z / db_.BLECapacity()];
        ble.ff[inst_z % db_.BLECapacity()] = inst_id;
        ble.numPins += 2;

        // Get net index of CK/SR/CE
        IndexType ck = kIndexTypeMax;
        IndexType sr = kIndexTypeMax;
        IndexType ce = kIndexTypeMax;
        for (IndexType j = 0; j < db_.instPins().size2(inst_id); ++j) {
          auto pin_id = db_.instPins().at(inst_id, j);
          switch (db_.pinSignalType(pin_id)) {
            case SignalType::kClock:
              ck = db_.pin2Net(pin_id);
              break;
            case SignalType::kControlSR:
              sr = db_.pin2Net(pin_id);
              break;
            case SignalType::kControlCE:
              ce = db_.pin2Net(pin_id);
              break;
            default:
              break;
          }
        }
        IndexType oe = inst_z % 2;
        ble.ck       = (ble.ck == kIndexTypeMax ? ck : ble.ck);
        ble.sr       = (ble.sr == kIndexTypeMax ? sr : ble.sr);
        ble.ce[oe]   = (ble.ce[oe] == kIndexTypeMax ? ce : ble.ce[oe]);
      }
    }

    // Optimize each CLB
    for (auto &clb : clbMap) {
      // Cache BLE original order in the CLB
      for (IndexType i = 0; i < db_.numBLEsPerCLB(); ++i) {
        clb.ble[i].origIdx = i;
      }
      optimizePinAccess(clb);
      optimizeBLEAlignment(clb);

      // Realize the solution
      for (IndexType bleIdx = 0; bleIdx < db_.numBLEsPerCLB(); ++bleIdx) {
        const auto &b   = clb.ble[bleIdx];
        IndexType   pos = bleIdx * db_.BLECapacity();
        for (IndexType k = 0; k < db_.BLECapacity(); ++k) {
          if (b.lut[k] != kIndexTypeMax) {
            state_.pos[b.lut[k] * 3 + 2] = pos + k;
          }
          if (b.ff[k] != kIndexTypeMax) {
            state_.pos[b.ff[k] * 3 + 2] = pos + k;
          }
        }
      }
    }
    openparfPrint(kInfo, "Intra-CLB optimization done\n");
  }

  /// Optimize pin access
  /// In this function, we reassign LUT/FF in each slice to better spread pins
  void optimizePinAccess(CLB &clb) const {
    // Alias
    auto &                   ble = clb.ble;

    // Get the CLB control set
    std::array<IndexType, 2> ck;
    std::array<IndexType, 2> sr;
    std::array<IndexType, 4> ce;
    clb.getCtrlSet(ck, sr, ce);

    // Here we first align the control set of the two half CLB if they are compatible
    bool compatible = false;
    if (ck[0] == kIndexTypeMax || ck[1] == kIndexTypeMax) {
      // One of the half CLB is empty, so the two half CLBs are control set compatible
      compatible = true;
    } else if (ck[0] == ck[1] && sr[0] == sr[1]) {
      // In this case, both half CLBs are not empty and they have the same CK/SR
      // We need to check their CE compatibility
      if ((ce[0] == kIndexTypeMax || ce[2] == kIndexTypeMax || ce[0] == ce[2]) &&
          (ce[1] == kIndexTypeMax || ce[3] == kIndexTypeMax || ce[1] == ce[3])) {
        // The 2 CE sets are directly compatible
        compatible = true;
      } else if ((ce[0] == kIndexTypeMax || ce[3] == kIndexTypeMax || ce[0] == ce[3]) &&
                 (ce[1] == kIndexTypeMax || ce[2] == kIndexTypeMax || ce[1] == ce[2])) {
        // The 2 CE sets are compatible if we filp one of them
        std::swap(ce[2], ce[3]);
        for (IndexType i = db_.numBLEsPerHalfCLB(); i < db_.numBLEsPerCLB(); ++i) {
          std::swap(ble[i].ff[0], ble[i].ff[1]);
          std::swap(ble[i].ce[0], ble[i].ce[1]);
        }
        compatible = true;
      }
    }

    // If the two half CLBs have compatible control sets, we perform pin access optimization over the whole CLB
    // If we have the 8 BLEs sorted by their pin counts from high to low denoted by (0, 1, 2, 3, 4, 5, 6, 7)
    // Then, we put them into the following order to interleave high-pin BLEs and low-pin BLEs to optimize pin access
    //   BLE index   0  1  2  3  4  5  6  7
    //              (0, 4, 2, 6, 1, 5, 3, 7)
    //
    // If the two half CLBs do not have compatible control sets, we perform pin access optimization within each half CLB
    // If we have the 4 BLEs sorted by their pin counts from high to low denoted by (0, 1, 2, 3)
    // Then, we put them into the following order to interleave high-pin BLEs and low-pin BLEs to optimize pin access
    //   BLE index   0  1  2  3
    //              (0, 2, 1, 3)
    //
    // If two BLEs have the same pin count, we preserve their original order
    // Besides distributing pin counts, we also want to preserve the initial order, so we also have a cost for it in the
    // sorting function
    auto comp = [](const typename CLB::BLE &a, const typename CLB::BLE &b) {
      return a.numPins + b.origIdx > b.numPins + a.origIdx;
    };
    std::vector<typename CLB::BLE> sortedBLE(ble);
    if (compatible) {
      std::sort(sortedBLE.begin(), sortedBLE.end(), comp);
      ble[0] = sortedBLE[0];
      ble[1] = sortedBLE[4];
      ble[2] = sortedBLE[2];
      ble[3] = sortedBLE[6];
      ble[4] = sortedBLE[1];
      ble[5] = sortedBLE[5];
      ble[6] = sortedBLE[3];
      ble[7] = sortedBLE[7];
    } else {
      for (IndexType offset : {0, static_cast<IndexType>(db_.numBLEsPerHalfCLB())}) {
        std::sort(sortedBLE.begin() + offset, sortedBLE.begin() + offset + db_.numBLEsPerHalfCLB(), comp);
        ble[offset + 0] = sortedBLE[offset + 0];
        ble[offset + 1] = sortedBLE[offset + 2];
        ble[offset + 2] = sortedBLE[offset + 1];
        ble[offset + 3] = sortedBLE[offset + 3];
      }
    }
  }

  /// Optimize BLE alignment
  /// We align LUT-FF pairs if they have internel nets or strong connection
  void optimizeBLEAlignment(CLB &clb) const {
    // We perform BLE alignment for each half CLB
    // Note that, if there are two different CEs in a half CLB, all FF pairs must stay/swap together
    // Since FFs in different BLE may have different preferred directions (stay/swap),
    // we let them vote and take the decision make by majority in this case
    //
    // If there is only one CE (or the two CEs are the same) in a half CLB, each BLE can make their moves independently
    //
    // pf[i] is the preferred choice of the i-th BLE
    //   pf[i] > 0: prefer to 'stay'
    //   pf[i] < 0: prefer to 'swap'
    //   |pf[i]| represents the stregth of the choice

    // Compute perference of each BLE
    std::vector<RealType> pf(db_.numBLEsPerCLB());
    // std::array<RealType, db_.numBLEsPerCLB()> pf;
    for (IndexType i = 0; i < db_.numBLEsPerCLB(); ++i) {
      pf[i] = computeBLEAlignmentPreference(clb.ble[i]);
    }

    // Get the CLB control set
    std::array<IndexType, 2> ck;
    std::array<IndexType, 2> sr;
    std::array<IndexType, 4> ce;
    clb.getCtrlSet(ck, sr, ce);

    for (IndexType lh : {0, 1}) {
      IndexType beg   = lh * db_.numBLEsPerHalfCLB();
      IndexType end   = beg + db_.numBLEsPerHalfCLB();
      IndexType ceIdx = lh * 2;
      if (ce[ceIdx] == kIndexTypeMax || ce[ceIdx + 1] == kIndexTypeMax || ce[ceIdx] == ce[ceIdx + 1]) {
        // BLEs in this half CLB can make their moves independently
        for (IndexType i = beg; i < end; ++i) {
          if (pf[i] < 0) {
            std::swap(clb.ble[i].ff[0], clb.ble[i].ff[1]);
          }
        }
      } else if (std::accumulate(pf.begin() + beg, pf.begin() + end, 0.0) < 0.0) {
        // BLEs in this half CLB must take move together
        for (IndexType i = beg; i < end; ++i) {
          std::swap(clb.ble[i].ff[0], clb.ble[i].ff[1]);
        }
      }
    }
  }

  /// Compute BLE alignment preference for BLE alignment optimization
  /// @return  the preferred choice of the i-th BLE
  ///           ret > 0: prefer to 'stay'
  ///           ret < 1: prefer to 'swap'
  ///            |ret| : the strength of the choice
  RealType computeBLEAlignmentPreference(const typename CLB::BLE &ble) const {
    RealType res = 0;
    for (IndexType k : {0, 1}) {
      if (ble.lut[k] == kIndexTypeMax) {
        continue;
      }
      // Get the output pin of this LUT
      IndexType lut    = ble.lut[k];
      IndexType outNet = kIndexTypeMax;
      for (IndexType j = 0; j < db_.instPins().size2(lut); ++j) {
        auto pin_id = db_.instPins().at(lut, j);
        if (db_.pinSignalDirect(pin_id) == SignalDirection::kOutput) {
          outNet = db_.pin2Net(pin_id);
          break;
        }
      }
      if (outNet == kIndexTypeMax) {
        // This LUT is not connected to any output pin
        continue;
      }
      // Find FFs in this outNet such that
      //   (1) the incident pin is an input pin
      //   (2) the FF is also in this BLE
      auto net_num_pins = db_.netPins().size2(outNet);
      for (IndexType j = 0; j < net_num_pins; ++j) {
        auto pin_id = db_.netPins().at(outNet, j);
        if (db_.pinSignalDirect(pin_id) == SignalDirection::kInput) {
          if (db_.pin2Inst(pin_id) == ble.ff[k]) {
            res += 1.0 / (net_num_pins - 1);
          }
          if (db_.pin2Inst(pin_id) == ble.ff[1 - k]) {
            res -= 1.0 / (net_num_pins - 1);
          }
        }
      }
    }
    return res;
  }

  database::PlaceDB const &                         db_;
  ISMDetailedPlaceParam                             param_;
  std::function<bool(uint32_t, uint32_t, uint32_t)> isCKAllowedInSite_;
  int32_t                                           num_threads_;
  bool                                              honorClockConstraint_;

  ISMDetailedPlaceState                             state_;   ///< intermediate state
};
}   // namespace ism_dp
OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DETAILED_PLACER_HPP_
