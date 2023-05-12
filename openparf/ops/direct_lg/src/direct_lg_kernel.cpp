/**
 * File              : direct_lg_kernel.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ops/direct_lg/src/direct_lg_kernel.h"

// C++ system libraries headers
#include <algorithm>
#include <string>
#include <vector>

// project headers
#include "database/placedb.h"

OPENPARF_BEGIN_NAMESPACE

namespace direct_lg {

/// Initialize the netlist information
void initDLProblemNetlist(database::PlaceDB const &db, DLProblem &prob) {
  RealType   avgLUTArea = 0;
  RealType   avgFFArea  = 0;
  IndexType &numLUT     = (prob.numLUTInst = 0);
  IndexType &numFF      = (prob.numFFInst = 0);
  IndexType  lram_count = 0, shift_count = 0, lut_count = 0;

  prob.pinToInst.assign(db.pin2Inst().begin(), db.pin2Inst().end());
  prob.pinToNet.assign(db.pin2Net().begin(), db.pin2Net().end());
  prob.pinTypes.resize(db.numPins());
  prob.netWts.assign(db.netWeights().begin(), db.netWeights().end());
  prob.instTypes.resize(db.numInsts());
  prob.instDemands.assign(db.numInsts(), 0);
  prob.instWts.assign(db.numInsts(), 0);
  prob.instAreas.assign(db.numInsts(), 0.0);
  prob.isInstFixed.resize(db.numInsts(), false);

  // Collect netlist connectivity
  for (uint32_t pid = 0; pid < db.numPins(); ++pid) {
    auto  signal_direct = db.pinSignalDirect(pid);
    auto  signal_type   = db.pinSignalType(pid);
    auto &ptype         = prob.pinTypes[pid];
    switch (signal_type) {
      case SignalType::kCascade:
      case SignalType::kSignal:
        if (signal_direct == SignalDirection::kInput) {
          ptype = DLPinType::INPUT;
        } else {
          ptype = DLPinType::OUTPUT;
        }
        break;
      case SignalType::kClock:
        ptype = DLPinType::CK;
        break;
      case SignalType::kControlSR:
        ptype = DLPinType::SR;
        break;
      case SignalType::kControlCE:
        ptype = DLPinType::CE;
        break;
      default:
        openparfAssert(0);
    }
  }

  // Compute average LUT/FF area, do not handle LRAM/SHIFT yet
  for (uint32_t i = 0; i < db.numInsts(); ++i) {
    auto const &ats  = db.instAreaType(i);
    RealType    area = 0;
    for (auto at : ats) {
      area = std::max(area, db.instSize(i, at).area());
    }
    if (db.isInstLUT(i)) {
      avgLUTArea += area;
      ++numLUT;
    } else if (db.isInstFF(i)) {
      avgFFArea += area;
      ++numFF;
    }
  }
  avgLUTArea /= numLUT;
  avgFFArea /= numFF;

  // Collect instance locations, instance types, demands, and weights
  for (uint32_t i = 0; i < db.numInsts(); ++i) {
    auto const &ats  = db.instAreaType(i);
    RealType    area = 0;
    for (auto at : ats) {
      area = std::max(area, db.instSize(i, at).area());
    }
    if (db.isInstLUT(i)) {
      IndexType modelId = db.getInstModelId(i);
      // TODO(jingmai@pku.edu.cn): magic number
      if (prob.useXarchLgRule && modelId == 4) {
        prob.instTypes[i] = DLInstanceType::LRAM;
        lram_count++;
        // TODO(jingmai@pku.edu.cn): magic number
      } else if (prob.useXarchLgRule && modelId == 5) {
        prob.instTypes[i] = DLInstanceType::SHIFT;
        shift_count++;
      } else {
        prob.instTypes[i] = DLInstanceType::LUT;
        lut_count++;
      }
      prob.instDemands[i] = db.isInstLUT(i);
      prob.instAreas[i]   = area / avgLUTArea;
    } else if (db.isInstFF(i)) {
      prob.instTypes[i] = DLInstanceType::FF;
      prob.instAreas[i] = area / avgFFArea;
    } else {
      if (i >= db.fixedRange().first && i < db.fixedRange().second) {
        prob.isInstFixed[i] = true;
      }
      prob.instTypes[i] = DLInstanceType::DONTCARE;
    }
  }

  // Compute the instance weight
  for (IndexType pid = 0; pid < prob.pinToNet.size(); ++pid) {
    prob.instWts[prob.pinToInst[pid]] += prob.netWts[prob.pinToNet[pid]];
  }

  // Initialize pin offsets
  // For LUT/FF/IO the pin offsets are (0, 0)
  // For DSP/RAM the pin offsets are 0.5 * instance width/height
  // 1. we assume pins are at center
  // 2. we are given center positions of instances
  prob.pinOffsets.resize(db.numPins());
  for (uint32_t i = 0; i < db.numPins(); ++i) {
    prob.pinOffsets[i] = {0, 0};
  }

  if (prob.useXarchLgRule) {
    openparfPrint(kDebug, "LUT %i, LRAM %i, SHIFT %i \n", lut_count, lram_count, shift_count);
  }
}

/// Initialize the site map information for the DL problem
void initDLProblemSiteMap(database::PlaceDB const &db, DLProblem &prob) {
  auto const &layout   = db.db()->layout();
  IndexType & numSLICE = (prob.numSiteSLICE = 0);
  IndexType & num_LUTs = (prob.num_LUTs = 0);
  IndexType & num_FFs  = (prob.num_FFs = 0);

  prob.siteTypes.resize(db.siteMapDim().x(), db.siteMapDim().y());
  prob.siteXYs.resize(db.siteMapDim().x(), db.siteMapDim().y());

  // initialize all sites
  // site_map does not contain duplicated sites
  for (uint32_t x = 0; x < db.siteMapDim().x(); ++x) {
    for (uint32_t y = 0; y < db.siteMapDim().y(); ++y) {
      prob.siteTypes(x, y) = DLSiteType::DONTCARE;
      prob.siteXYs(x, y).set(x, y);
    }
  }

  // We assume all instances should be placed at the center in a slice
  // So we have SLICE offset (0.5, 0.5)
  for (auto const &site : layout.siteMap()) {
    auto x = site.bbox().xl();
    auto y = site.bbox().yl();
    for (auto const &resource : layout.resourceMap()) {
      auto        site_type = layout.siteType(site);
      std::string site_name = site_type.name();
      if (prob.useXarchLgRule) {
        /* xarch Benchmark */
        if (site_name == "FUAT" || site_name == "FUBT") {
          if (site_name == "FUBT") {
            prob.siteTypes(x, y) = DLSiteType::SLICEM;
          } else {
            prob.siteTypes(x, y) = DLSiteType::SLICE;
          }
          numSLICE++;
          prob.siteXYs(x, y).set(x + 0.5, y + 0.5);
          if (resource.name() == "LUTL") {
            num_LUTs = std::max(num_LUTs, site_type.resourceCapacity(resource.id()));
          }
          if (resource.name() == "LUTM") {
            num_FFs = std::max(num_FFs, site_type.resourceCapacity(resource.id()));
          }
        }
      } else {
        /* ISPD benchmark */
        if (db.isResourceLUT(resource.id()) && site_type.resourceCapacity(resource.id())) {
          prob.siteTypes(x, y) = DLSiteType::SLICE;
          numSLICE++;
          prob.siteXYs(x, y).set(x + 0.5, y + 0.5);
          num_LUTs = std::max(num_LUTs, site_type.resourceCapacity(resource.id()));
        }
        if (db.isResourceFF(resource.id()) && site_type.resourceCapacity(resource.id())) {
          num_FFs = std::max(num_FFs, site_type.resourceCapacity(resource.id()));
        }
      }
    }
  }
  openparfPrint(kDebug, "#CLB-SLICE: %d, #LUTs per Site: %d, #FFs per Site: %d\n", numSLICE, num_LUTs, num_FFs);
}

/// Initialize the DL problem
void initDLProblem(database::PlaceDB const &db, DLProblem &prob) {
  // Initialize netlist information
  initDLProblemNetlist(db, prob);
  // Initialize the site map information
  initDLProblemSiteMap(db, prob);
}

void writeHalfColumnAvailabilityMapSolution(database::PlaceDB const &db, DLSolver const &solver, uint8_t *hcAvailMap) {
  for (int32_t hc_id = 0; hc_id < db.numHalfColumnRegions(); hc_id++) {
    for (int32_t clk = 0; clk < db.numClockNets(); clk++) {
      hcAvailMap[clk * db.numHalfColumnRegions() + hc_id] = solver.clkAvailableAtHalfColumn(clk, hc_id);
    }
  }
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void OPENPARF_NOINLINE directLegalizeLauncher<T>(                                                           \
          database::PlaceDB const &                db,                                                                 \
          DirectLegalizeParam const &              param,                                                              \
          T const *                                init_pos,                                                           \
          ClockAvailCheckerType<T>                 legality_check_functor,                                             \
          LayoutXy2GridIndexFunctorType<T>         xy_to_half_column_functor,                                          \
          const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,                                              \
          int32_t                                  num_threads,                                                        \
          T *                                      pos,                                                                \
          uint8_t *                                hc_avail_map);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace direct_lg

OPENPARF_END_NAMESPACE
