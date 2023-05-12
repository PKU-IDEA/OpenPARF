/**
 * File              : dl_solver.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 05.19.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "ops/direct_lg/src/dl_solver.h"

// C++ standard libray headers
#include <algorithm>
#include <map>
#include <utility>

// other headers
#include "lemon/matching.h"

OPENPARF_BEGIN_NAMESPACE

namespace direct_lg {

DLSolver::DLSolver(DLProblem const                         &prob,
                   const DirectLegalizeParam               &param,
                   ClockAvailCheckerType<RealType>          legality_check_functor,
                   LayoutXy2GridIndexFunctorType<RealType>  xy_to_half_column_functor,
                   const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
                   IndexType                                num_threads)
    : _param(param),
      _spiralAccessor(std::max(prob.siteTypes.xSize(), prob.siteTypes.ySize())),
      _bndBox(0, 0, prob.siteTypes.xSize() - 1, prob.siteTypes.ySize() - 1),
      _legality_check_functor(legality_check_functor),
      _xy_to_half_column_functor(xy_to_half_column_functor),
      _instClockIndex(inst_to_clock_indexes),
      _num_threads(num_threads) {
  buildDLProblem(prob);
}

void DLSolver::buildDLProblem(DLProblem const &prob) {
  // Build and sort the netlist
  buildNetlist(prob);
  sortNetlist();

  // Build the site map
  buildSiteMap(prob);

  // Initialize control sets
  initControlSets();

  // Initialize clock related data structures.
  if (_param.honorClockRegionConstraint) {
    initClockConstraints();
  }
}

/// Build the netlist
void DLSolver::buildNetlist(DLProblem const &prob) {
  // Construct instances and set their locations, types, and demands
  _instArray.clear();
  _instArray.resize(prob.instXYs.size());
  for (IndexType i = 0; i < _instArray.size(); ++i) {
    auto &inst   = _instArray[i];
    inst.origId  = i;
    inst.id      = i;
    inst.loc     = prob.instXYs[i];
    inst.type    = prob.instTypes[i];
    inst.dem     = prob.instDemands[i];
    inst.weight  = prob.instWts[i];
    inst.area    = prob.instAreas[i];
    inst.isFixed = prob.isInstFixed[i];
  }
  _numLUTInst = prob.numLUTInst;
  _numFFInst  = prob.numFFInst;
  // Construct nets and set their weights
  _netArray.clear();
  _netArray.resize(prob.netWts.size());
  for (IndexType i = 0; i < _netArray.size(); ++i) {
    auto &net  = _netArray[i];
    net.origId = i;
    net.id     = i;
    net.weight = prob.netWts[i];
  }

  // Construct pins and set their offsets, instance IDs and net IDs
  // We also need to add pin IDs into their corresponding instances and nets
  _pinArray.clear();
  _pinArray.resize(prob.pinOffsets.size());
  for (IndexType i = 0; i < _pinArray.size(); ++i) {
    auto &pin  = _pinArray[i];
    pin.origId = i;
    pin.offset = prob.pinOffsets[i];
    pin.type   = prob.pinTypes[i];
    pin.instId = prob.pinToInst[i];
    pin.netId  = prob.pinToNet[i];
    _instArray[pin.instId].pinIdArray.push_back(i);
    _netArray[pin.netId].pinIdArray.push_back(i);
  }
}

/// Sort instance, net, and pin objects in the netlist
/// The sorted netlist should have the following properties
///   1) nets are sorted by their degrees (number of pins) from low to high
///   2) pins are sorted by the IDs of their incident nets from low to high
///   3) instances are sorted by their smallest IDs of their incident pins (also nets, since pins are sorted by net IDs)
///   from low to high 4) pin IDs in inst.pinIdArray and net.pinIdArray are sorted from low to high, which also refect
///   their net ordering
void DLSolver::sortNetlist() {
  // Sort nets
  auto netComp = [&](IndexType a, IndexType b) {
    const auto &netA = _netArray[a];
    const auto &netB = _netArray[b];
    return (netA.numPins() == netB.numPins() ? netA.origId < netB.origId : netA.numPins() < netB.numPins());
  };
  sortNets(netComp);

  // Sort pins
  auto pinComp = [&](IndexType a, IndexType b) {
    const auto &pinA = _pinArray[a];
    const auto &pinB = _pinArray[b];
    return (pinA.netId == pinB.netId ? pinA.origId < pinB.origId : pinA.netId < pinB.netId);
  };
  sortPins(pinComp);

  // Sort instances
  auto instComp = [&](IndexType a, IndexType b) {
    const auto &instA = _instArray[a];
    const auto &instB = _instArray[b];
    if (instA.numPins() == 0) {
      return false;
    }
    if (instB.numPins() == 0) {
      return true;
    }
    return instA.pinIdArray[0] < instB.pinIdArray[0];
  };
  sortInstances(instComp);
}

/// Sort nets using the given comparator
void DLSolver::sortNets(std::function<bool(IndexType, IndexType)> cmp) {
  // Get net ordering and ranking
  IndexVector netOrder(numNets());
  IndexVector netRank(numNets());
  std::iota(netOrder.begin(), netOrder.end(), 0);
  std::sort(netOrder.begin(), netOrder.end(), cmp);
  for (IndexType rank = 0; rank < numNets(); ++rank) {
    netRank[netOrder[rank]] = rank;
  }

  // Construct the new netArray
  std::vector<DLNet> netArray(numNets());
  for (IndexType id = 0; id < numNets(); ++id) {
    netArray[netRank[id]] = _netArray[id];
  }
  _netArray = std::move(netArray);

  // Update netId in nets
  for (IndexType i = 0; i < numNets(); ++i) {
    _netArray[i].id = i;
  }
  // Update netId in pins
  for (auto &pin : _pinArray) {
    pin.netId = netRank[pin.netId];
  }
}

/// Sort pins using the given comparator
void DLSolver::sortPins(std::function<bool(IndexType, IndexType)> cmp) {
  // Get pin ordering and ranking
  IndexVector pinOrder(numPins());
  IndexVector pinRank(numPins());
  std::iota(pinOrder.begin(), pinOrder.end(), 0);
  std::sort(pinOrder.begin(), pinOrder.end(), cmp);
  for (IndexType rank = 0; rank < numPins(); ++rank) {
    pinRank[pinOrder[rank]] = rank;
  }

  // Construct the new pinArray
  std::vector<DLPin> pinArray(numPins());
  for (IndexType id = 0; id < numPins(); ++id) {
    pinArray[pinRank[id]] = _pinArray[id];
  }
  _pinArray = std::move(pinArray);

  // Update pinId in nets and instances, and also sort them
  for (auto &net : _netArray) {
    for (IndexType &pinId : net.pinIdArray) {
      pinId = pinRank[pinId];
    }
    std::sort(net.pinIdArray.begin(), net.pinIdArray.end());
  }
  for (auto &inst : _instArray) {
    for (IndexType &pinId : inst.pinIdArray) {
      pinId = pinRank[pinId];
    }
    std::sort(inst.pinIdArray.begin(), inst.pinIdArray.end());
  }
}

/// Sort instances using the given comparator
void DLSolver::sortInstances(std::function<bool(IndexType, IndexType)> cmp) {
  // Get instance ordering and ranking
  IndexVector instOrder(numInsts());
  std::iota(instOrder.begin(), instOrder.end(), 0);
  std::sort(instOrder.begin(), instOrder.end(), cmp);

  auto &instRank = _instIdMapping;
  instRank.resize(numInsts());
  for (IndexType rank = 0; rank < numInsts(); ++rank) {
    instRank[instOrder[rank]] = rank;
  }

  // Construct the new instArray
  std::vector<DLInstance> instArray(numInsts());
  for (IndexType id = 0; id < numInsts(); ++id) {
    instArray[instRank[id]] = _instArray[id];
  }
  _instArray = std::move(instArray);

  // Update instId in instances
  for (IndexType i = 0; i < numInsts(); ++i) {
    _instArray[i].id = i;
  }
  // Update instId in pins
  for (auto &pin : _pinArray) {
    pin.instId = instRank[pin.instId];
  }
}

/// Build the site map
void DLSolver::buildSiteMap(DLProblem const &prob) {
  _num_LUTs = prob.num_LUTs;
  _num_FFs  = prob.num_FFs;
  _siteMap.resize(prob.siteTypes.sizes(), DLSite(_param.candPQSize, _num_LUTs, _num_FFs));
  for (IndexType i = 0; i < _siteMap.size(); ++i) {
    auto &site = _siteMap[i];
    site.id    = i;
    site.type  = prob.siteTypes[i];
    site.loc   = prob.siteXYs[i];
  }
  _numSiteSLICE = prob.numSiteSLICE;
}

/// Initialize the control sets in FF
void DLSolver::initControlSets() {
  _cksrMapping.clear();
  _ceMapping.clear();
  IndexType cksrId = 0;
  IndexType ceId   = 0;

  for (auto &inst : _instArray) {
    if (inst.type != DLInstanceType::FF) {
      inst.cksr = inst.ce = kIndexTypeMax;
      continue;
    }
    // Find the net ID of CK, CE, and SR pins
    IndexType ck = kIndexTypeMax, sr = kIndexTypeMax, ce = kIndexTypeMax;
    for (IndexType pinId : inst.pinIdArray) {
      const auto &pin = _pinArray[pinId];
      switch (pin.type) {
        case DLPinType::CK:
          ck = pin.netId;
          break;
        case DLPinType::SR:
          sr = pin.netId;
          break;
        case DLPinType::CE:
          ce = pin.netId;
          break;
        default: {
          // do nothing
        }
      }
    }

    // Set the CK/SR mapping
    auto ckIt = _cksrMapping.find(ck);
    if (ckIt == _cksrMapping.end()) {
      inst.cksr            = cksrId;
      _cksrMapping[ck][sr] = cksrId++;
    } else {
      auto &srMap = ckIt->second;
      auto  srIt  = srMap.find(sr);
      if (srIt == srMap.end()) {
        inst.cksr = cksrId;
        srMap[sr] = cksrId++;
      } else {
        inst.cksr = srIt->second;
      }
    }

    // Set the CE mapping
    auto ceIt = _ceMapping.find(ce);
    if (ceIt == _ceMapping.end()) {
      inst.ce        = ceId;
      _ceMapping[ce] = ceId++;
    } else {
      inst.ce = ceIt->second;
    }
  }
}

void DLSolver::initClockConstraints() {
  openparfPrint(kDebug, "%i clock nets, %i half columns\n", _param.numClockNet, _param.numHalfColumn);
  for (IndexType i = 0; i < _param.numHalfColumn; i++) {
    _hcsbArray.emplace_back(_param.numClockNet);
  }
  // Make all clock available at first
  for (auto &sb : _hcsbArray) {
    std::fill(sb.isAvail.begin(), sb.isAvail.end(), 1);
  }

  std::vector<uint32_t> fixed_clk_per_hc(_param.numHalfColumn, 0);
  // We must alway consider the clocks of the DONTCARE instances in the halfcolumn scoreboard, like DSPs and RAMs.
  // These instances must be placed with in one half column as well as one clock region.
  for (auto const &inst : _instArray) {
    if (inst.type == DLInstanceType::DONTCARE && !inst.isFixed) {
      auto  hc_id = _xy_to_half_column_functor(inst.x(), inst.y());
      auto &sb    = _hcsbArray.at(hc_id);
      for (auto &clk : _instClockIndex[inst.origId]) {
        if (sb.isFixed.at(clk) != 1) fixed_clk_per_hc.at(hc_id)++;
        sb.isFixed.at(clk) = 1;
        openparfAssertMsg(fixed_clk_per_hc.at(hc_id) <= _param.maxClockNetPerHalfColumn,
                          "DONTCARE instances already have %i clock nets in half column %i",
                          fixed_clk_per_hc.at(hc_id),
                          hc_id);
      }
    }
  }
}

void DLSolver::calculateHalfColumnScoreBoardCost() {
  if (!_param.honorHalfColumnConstraint) {
    return;
  }
  for (auto &sb : _hcsbArray) {
    std::fill(sb.clockCost.begin(), sb.clockCost.end(), 0);
  }
  RealType baseCost = (RealType) _siteMap.size() / (RealType) _numSiteSLICE;
  openparfPrint(kDebug, "baseCost is: %f\n", baseCost);
  for (auto &inst : _instArray) {
    if (inst.type == DLInstanceType::DONTCARE) continue;
    openparfAssertMsg(inst.type == DLInstanceType::LUT || inst.type == DLInstanceType::FF,
                      "Only LUT or FF considered at the moment");
    openparfAssertMsg(inst.curr.detSiteId != kIndexTypeMax,
                      "half column scoreboard cost should only be calculated when all inst had been found a location.");
    // Since we only considers LUT and FF, there's actually no need to set a cost like this
    // I'm leaving it here so that we can expand direct_lg to accommodate for other resources
    RealType cost   = baseCost * inst.numPins();

    auto     hc_idx = _xy_to_half_column_functor(_siteMap[inst.curr.detSiteId].x(), _siteMap[inst.curr.detSiteId].y());
    auto    &sb     = _hcsbArray.at(hc_idx);
    for (auto &clk : _instClockIndex[inst.origId]) {
      sb.clockCost.at(clk) += cost;
    }
  }
}

// Check whether the half column constraint is satisfied.
// While doing so, prohibit the least costly clock in every half column region that
// does not satisfy the half column constraint.
bool DLSolver::checkHalfColumnSatAndPruneClock() {
  bool isLegal = true;
  for (IndexType hcId = 0; hcId < _hcsbArray.size(); hcId++) {
    auto     &sb           = _hcsbArray.at(hcId);
    // count the number of clk in this hc
    IndexType cnt          = 0;
    IndexType pruneClockId = kIndexTypeMax;
    IndexType minCost      = kIndexTypeMax;
    for (IndexType i = 0; i < sb.clockCost.size(); i++) {
      openparfAssertMsg(sb.isAvail[i] || (std::abs(sb.clockCost[i]) < 1e-6),
                        "Clk %i is in hc %i when it's not available.",
                        i,
                        hcId);
      if (sb.isFixed[i] || sb.clockCost[i] > 1e-6) {
        cnt++;
      }
      if (!sb.isFixed[i] && sb.clockCost[i] > 1e-6) {
        if (sb.clockCost[i] < minCost) {
          minCost      = sb.clockCost[i];
          pruneClockId = i;
        }
      }
    }
    if (cnt <= _param.maxClockNetPerHalfColumn) continue;
    openparfAssertMsg(pruneClockId != kIndexTypeMax,
                      "All clock are fixed and none can be pruned. This should not happen.");
    // If clock number exceeds what is allowed, we disable the clock with the lowest non zero cost
    openparfPrint(kDebug, "prune clk %i in hc region %i\n", pruneClockId, hcId);
    sb.isAvail.at(pruneClockId) = 0;
    isLegal                     = false;
  }
  return isLegal;
}


/// Top function to perform the direct legalization
void DLSolver::run() {
  DLStatus status;

  for (auto &instance : _instArray) {
    instance.initLock();
  }
  // Initialization
  initNets();
  preclustering();
  initSiteNeighbors();
  allocateMemory();

  // Perform DL kernel
  runDLInit();

  printDLStatus();
  status = DLStatus::ACTIVE;
  while (status == DLStatus::ACTIVE) {
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, _param.omp_dynamic_chunk_size)
    for (IndexType siteId = 0; siteId < numSites(); ++siteId) {
      runDLIteration(_siteMap[siteId]);
    }
    status = runDLSync();

    ++_dlIter;
    printDLStatus();
  }
  printDLStatistics();

  // Iteratively perform post legalization to legalize remaining instances & to conform with
  // clock region constraints. Also conform with hc constraints by using a scoreboard relaxation
  // method.
  do {
    if (!ripupLegalization()) {
      openparfAssertMsg(false, "Legalization fails");
    }
    calculateHalfColumnScoreBoardCost();
  } while (!checkHalfColumnSatAndPruneClock());

  printPostLegalizationStatistics();
  // Perform slot assignment
  runSlotAssign();

  cacheSolution();

  for (auto &instance : _instArray) {
    instance.destroyLock();
  }
}

/// Initialize net information
void DLSolver::initNets() {
  for (IndexType i = 0; i < numNets(); ++i) {
    auto &net = _netArray[i];
    if (net.numPins() == 0 || net.numPins() > _param.wirelenScoreMaxNetDegree) {
      net.bbox.set(0, 0, 0, 0);
      net.pinIdArrayX.clear();
      net.pinIdArrayY.clear();
      continue;
    }

    // Compute the net bounding box
    net.bbox.set(kRealTypeMax, kRealTypeMax, kRealTypeMin, kRealTypeMin);
    for (IndexType pinId : net.pinIdArray) {
      const auto &pin  = _pinArray[pinId];
      const auto &inst = _instArray[pin.instId];
      net.bbox.encompass(inst.loc + pin.offset);
    }

    // Compute the X sorted pin ID array
    net.pinIdArrayX = net.pinIdArray;
    auto compX      = [&](IndexType a, IndexType b) {
      const auto &pinA  = _pinArray[a];
      const auto &pinB  = _pinArray[b];
      const auto &instA = _instArray[pinA.instId];
      const auto &instB = _instArray[pinB.instId];
      return instA.x() + pinA.offsetX() < instB.x() + pinB.offsetX();
    };
    std::sort(net.pinIdArrayX.begin(), net.pinIdArrayX.end(), compX);

    // Compute the Y sorted pin ID array
    net.pinIdArrayY = net.pinIdArray;
    auto compY      = [&](IndexType a, IndexType b) {
      const auto &pinA  = _pinArray[a];
      const auto &pinB  = _pinArray[b];
      const auto &instA = _instArray[pinA.instId];
      const auto &instB = _instArray[pinB.instId];
      return instA.y() + pinA.offsetY() < instB.y() + pinB.offsetY();
    };
    std::sort(net.pinIdArrayY.begin(), net.pinIdArrayY.end(), compY);
  }
}

/// Perform preclustering
/// We pair LUT and FF that have internal connection
/// TODO: Do we need to pair LRAMs and SHIFTs?
void DLSolver::preclustering() {
  // Initialize the precluster of each instance as itself
  for (IndexType i = 0; i < numInsts(); ++i) {
    auto &inst = _instArray[i];
    if (inst.type != DLInstanceType::DONTCARE) {
      inst.precluster.clear();
      inst.precluster.push_back(i);
    }
  }

  IndexType numSize2 = 0, numSize3 = 0;
  for (IndexType instId = 0; instId < numInsts(); ++instId) {
    auto &inst = _instArray[instId];
    if (inst.type != DLInstanceType::LUT) {
      continue;
    }
    // Get the LUT output pin and net
    auto outPinIt = std::find_if(inst.pinIdArray.begin(), inst.pinIdArray.end(), [&](IndexType i) {
      return _pinArray[i].type == DLPinType::OUTPUT;
    });
    if (outPinIt == inst.pinIdArray.end()) {
      continue;
    }
    // Get all FF instances with input pins in this net and within the maximum preclustering distance
    const auto                &outPin = _pinArray[*outPinIt];
    const auto                &outNet = _netArray[outPin.netId];
    std::vector<IndexRealPair> ffs;
    for (IndexType pinId : outNet.pinIdArray) {
      const auto &outNetPin  = _pinArray[pinId];
      const auto &outNetInst = _instArray[outNetPin.instId];
      RealType    dist       = (inst.loc + outPin.offset).manhattanDistance(outNetInst.loc + outNetPin.offset);
      if (outNetPin.type == DLPinType::INPUT && outNetInst.type == DLInstanceType::FF &&
          dist <= _param.preclusteringMaxDist) {
        ffs.emplace_back(outNetInst.id, dist);
      }
    }
    if (ffs.empty()) {
      continue;
    }

    // Sort all FFs by their distances to the LUT from low to high
    // Find up to 2 FFs to form the precluster, for the case of 2 FFs, they must be able to fit into the same BLE
    std::sort(ffs.begin(), ffs.end(), [&](const IndexRealPair &a, const IndexRealPair &b) {
      return a.second < b.second;
    });
    inst.precluster.push_back(ffs[0].first);
    IndexType cksr = _instArray[ffs[0].first].cksr;
    auto      it   = std::find_if(ffs.begin() + 1, ffs.end(), [&](const IndexRealPair &p) {
      return _instArray[p.first].cksr == cksr;
    });
    if (it != ffs.end()) {
      inst.precluster.push_back(it->first);
    }

    // Keep precluster instance IDs sorted
    std::sort(inst.precluster.begin(), inst.precluster.end());
    for (IndexType i : inst.precluster) {
      _instArray[i].precluster = inst.precluster;
    }
    (inst.precluster.size() == 2 ? numSize2 : numSize3) += 1;
  }
  openparfPrint(kInfo, "# Precluster : %u (%u + %u)\n", numSize2 + numSize3, numSize2, numSize3);
}

/// Initialize site neighbor instances
/// We divide neighbor instances of each site into groups by their distance to the site
void DLSolver::initSiteNeighbors() {
  // Temporary map to store neighbor instances ID and distances for each site
  using NDPair = std::pair<IndexType, RealType>;
  Vector2D<std::vector<NDPair>> nbrListMap(_siteMap.sizes());

  // Add each instance to its neighbor sites through a spiral walk
  IndexType                     maxD = std::ceil(_param.nbrDistEnd) + 1;
  auto                          beg  = _spiralAccessor.begin(0);
  auto                          end  = _spiralAccessor.end(maxD);
  for (IndexType instId = 0; instId < numInsts(); ++instId) {
    // We skip DONTCARE instances and instances that is not the first one in their precluster
    const auto &inst = _instArray[instId];
    if (inst.type == DLInstanceType::DONTCARE || inst.precluster[0] != instId) {
      continue;
    }

    // Calculate the centroid of the precluster
    XY<RealType> initXY(computeInstCentroid(inst.precluster));
    for (auto it = beg; it != end; ++it) {
      XY<IntType> xy(IntType(initXY.x() + it->x()), IntType(initXY.y() + it->y()));
      if (!_bndBox.contain(xy)) {
        continue;
      }
      auto &site = _siteMap(xy.x(), xy.y());
      if (instAndSiteAreCompatible(inst, site) && instAndSiteClockCompatible(inst, site)) {
        RealType dist = initXY.manhattanDistance(site.loc);
        if (dist < _param.nbrDistEnd) {
          nbrListMap(xy.x(), xy.y()).emplace_back(instId, dist);
        }
      }
    }
  }

  // Sort neighbor instances in each site and divide them into groups
  IndexType numGroups = std::ceil((_param.nbrDistEnd - _param.nbrDistBeg) / _param.nbrDistIncr) + 1;
  for (IndexType i = 0; i < numSites(); ++i) {
    // Sort neighbors by their distances
    auto &list = nbrListMap[i];
    std::sort(list.begin(), list.end(), [&](const NDPair &l, const NDPair &r) { return l.second < r.second; });

    // Set group ranges
    auto &site = _siteMap[i];
    site.nbrList.clear();
    site.nbrList.reserve(list.size());
    if (!list.empty()) {
      site.nbrRanges.resize(numGroups + 1);
      site.nbrRanges[0]  = 0;
      IndexType groupIdx = 0;
      RealType  maxD     = _param.nbrDistBeg;
      for (IndexType i = 0; i < list.size(); ++i) {
        site.nbrList.push_back(list[i].first);
        while (list[i].second >= maxD) {
          site.nbrRanges[++groupIdx] = i;
          maxD += _param.nbrDistIncr;
        }
      }
      while (++groupIdx <= numGroups) {
        site.nbrRanges[groupIdx] = site.nbrList.size();
      }
    }
  }
}

/// Allocate memory before a DL run
void DLSolver::allocateMemory() {
  _threadMemArray.resize(omp_get_max_threads());
}

/// Run DL initialization
void DLSolver::runDLInit() {
  // Initialize sites
  for (IndexType i = 0; i < numSites(); ++i) {
    auto &site = _siteMap[i];
    if (site.numNbrGroups() > 0) {
      site.nbr.assign(site.nbrList.begin() + site.nbrRanges[0], site.nbrList.begin() + site.nbrRanges[1]);
      site.nbrGroupIdx = 1;
    }
    if (site.type != DLSiteType::DONTCARE) {
      site.det        = std::move(Candidate(_num_LUTs, _num_FFs));
      site.det.siteId = site.id;
      site.curr       = std::move(DLSite::SyncData(_param.candPQSize));
      site.next       = std::move(DLSite::SyncData(_param.candPQSize));
      site.curr.scl.emplace_back(site.det);
    }
  }

  // Initialize instances
  for (IndexType i = 0; i < numInsts(); ++i) {
    auto &inst = _instArray[i];
    if (inst.type != DLInstanceType::DONTCARE) {
      inst.curr = std::move(DLInstance::SyncData());
      inst.next = std::move(DLInstance::SyncData());
    }
  }
  _dlIter = 0;
}

/// Run a DL iteration for a site
void DLSolver::runDLIteration(DLSite &site) {
  if (site.type == DLSiteType::DONTCARE) {
    return;
  }

  // We first try to commit the current best candidate
  if (!commitTopCandidate(site)) {
    // If the top candidate is not commited, we need to remove invalid candidates in the site
    // Since some instances can be committed to other sites in the previous iteration and make some candidate become
    // invalid
    removeInvalidCandidates(site);
  }

  // Remove neighbor instances that haven been commited to sites
  removeCommittedNeighbors(site);

  // Add more neighbor instances into the site if needed
  addNeighbors(site);

  // Create new candidates by merging neighbor instances (nbr) and seed candidates (scl)
  createNewCandidates(site);

  // Broadcast the top candidate to all constituting instances
  broadcastTopCandidate(site);
}

/// Commit the top candidate of a given site if possible
/// Return true if the top candidate is commited
bool DLSolver::commitTopCandidate(DLSite &site) {
  // We DO NOT commit the top candidate if one of the following holds
  //   1) there is no candidate in this site
  //   2) the top candidate has not been stable for a long enough time
  //   3) the top candidate is not valid
  if (site.curr.pq.empty() || site.curr.stable < _param.minStableIter || !candidateIsValid(site.curr.pq.top())) {
    return false;
  }

  // Commit the top candidate if all its constituting instances accept it
  const auto &top = site.curr.pq.top();
  for (IndexType instId : top.sig) {
    const auto &inst = _instArray[instId];
    if (inst.curr.detSiteId != site.id && inst.curr.bestSiteId != site.id) {
      return false;
    }
  }

  // If the execution hits here, we commit the top candidate
  // Also set the determined sites of all its constituting instances as this site
  site.det = top;
  for (IndexType instId : site.det.sig) {
    _instArray[instId].next.detSiteId = site.id;
  }

  // Remove all neighbor instances that are not compatible with the determined candidate
  removeIncompatibleNeighbors(site);

  // Clear pq and make the scl only contains the commited candidate
  site.next.pq.clear();
  site.curr.scl.clear();
  site.curr.scl.emplace_back(site.det);
  return true;
}

/// Given a site, remove all its neighbor instances that are not compatible with its determined candidate
void DLSolver::removeIncompatibleNeighbors(DLSite &site) {
  for (IndexType &instId : site.nbr) {
    // An instance can be incompatible if one of the following holds
    //   1) the instance has been added into the site (i.e., site.det contains it)
    //   2) the instance and the site's determined candidate (site.det) cannot form a legal candidate
    const auto &inst = _instArray[instId];
    if (std::find(site.det.sig.begin(), site.det.sig.end(), instId) != site.det.sig.end() ||
        !addInstToCandidateIsFeasible(inst, site.det)) {
      instId = kIndexTypeMax;
    }
  }
  // Remove invalid neighbor instance IDs
  site.nbr.erase(std::remove(site.nbr.begin(), site.nbr.end(), kIndexTypeMax), site.nbr.end());
}

/// Check whether adding an instance to a candidate is feasible
/// This function only perform the dry checking, it does not provide the feasible solution found
bool DLSolver::addInstToCandidateIsFeasible(const DLInstance &inst, const Candidate &cand) {
  Candidate::Implementation impl(_num_LUTs, _num_FFs);
  return addInstToCandidateImpl(inst, cand.impl, impl);
}

/// Add an instance into a candidate implementation, return true if it is legal
bool DLSolver::addInstToCandidateImpl(const DLInstance &inst, Candidate &cand) {
  Candidate::Implementation impl(_num_LUTs, _num_FFs);
  if (addInstToCandidateImpl(inst, cand.impl, impl)) {
    cand.impl = std::move(impl);
    return true;
  }
  return false;
}

/// Try to add an instance into a candidate implementation, return true on success
/// @param  instance  the instance to be added
/// @param  impl      the candidate implementation to be added
/// @param  res       the resulting candidate implementation if succeed
bool DLSolver::addInstToCandidateImpl(const DLInstance                &instance,
                                      const Candidate::Implementation &impl,
                                      Candidate::Implementation       &res) {
  res          = impl;
  bool lutFail = false;
  for (IndexType instId : instance.precluster) {
    const auto &inst = _instArray[instId];
    switch (inst.type) {
      case DLInstanceType::FF: {
        if (!addFFToCandidateImpl(inst, res)) {
          // If adding the FF fails, the whole adding fails
          return false;
        }
        break;
      }
      case DLInstanceType::LUT:
      case DLInstanceType::LRAM:
      case DLInstanceType::SHIFT: {
        if (!lutFail && !addLUTToCandidateImpl(inst, res)) {
          // If adding the LUT fails here, we can further perform max-matching based adding
          lutFail = true;
        }
        break;
      }
      default: {
        openparfAssertMsg(false, "Unexpected instance type");
      }
    }
  }
  if (!lutFail) {
    return true;
  }

  // Since there is no packing problem for LRAM and SHIFT,
  // if greedy adding does not work, then nothing would.
  if (res.candidate_type != LutCandidateType::UNDECIDED) {
    return false;
  }
  // If greedily adding LUTs fails, we perform more dedicated max-matching based algorithm
  auto &luts = _threadMemArray[omp_get_thread_num()].idxVecA;
  luts.assign(impl.lut.begin(), impl.lut.end());
  for (IndexType i : instance.precluster) {
    if (_instArray[i].type == DLInstanceType::LUT) {
      luts.push_back(i);
    }
  }
  return fitLUTsToCandidateImpl(luts, res);
}

/// Try to add an FF into a given candidate implementation , return true on success
bool DLSolver::addFFToCandidateImpl(const DLInstance &ff, Candidate::Implementation &impl) const {
  // i = 0/1 for the lower/upper half of the slice
  for (IndexType i : {0, 1}) {
    // Check CK/SR
    if (impl.cksr[i] != kIndexTypeMax && impl.cksr[i] != ff.cksr) {
      continue;
    }

    // j = 0/1 for the odd/even part of the half slice
    for (IndexType j : {0, 1}) {
      // Check CE
      if (impl.ce[2 * i + j] != kIndexTypeMax && impl.ce[2 * i + j] != ff.ce) {
        continue;
      }

      // CK/SR/CE are all compatible, check slot availability
      IndexType beg = i * _param.half_CLB_capacity + j;
      IndexType end = beg + _param.half_CLB_capacity;
      for (IndexType k = beg; k < end; k += _param.BLE_capacity) {
        if (impl.ff[k] == kIndexTypeMax) {
          // Found an empty slot, put the ff here and update control sets
          impl.ff[k]         = ff.id;
          impl.cksr[i]       = ff.cksr;
          impl.ce[2 * i + j] = ff.ce;
          return true;
        }
      }
    }
  }
  return false;
}

/// Try to add a LUT into a given candidate implementation using greedy method, return true on success
/// If it fails, feasible solution may still exist, but the max matching based method needs to be called
bool DLSolver::addLUTToCandidateImpl(const DLInstance &lut, Candidate::Implementation &impl) const {
  // Candidate implementation is good!
  // Check candidate type
  if (impl.candidate_type != LutCandidateType::UNDECIDED) {
    if ((impl.candidate_type == LutCandidateType::LRAM && lut.type != DLInstanceType::LRAM) ||
        (impl.candidate_type == LutCandidateType::SHIFT && lut.type != DLInstanceType::SHIFT) ||
        (impl.candidate_type == LutCandidateType::LUT && lut.type != DLInstanceType::LUT)) {
      return false;
    }
  }
  // First check the even slots
  for (IndexType i = 0; i < _param.CLB_capacity; i += _param.BLE_capacity) {
    if (impl.lut[i] == kIndexTypeMax) {
      impl.lut[i] = lut.id;
      if (impl.candidate_type == LutCandidateType::UNDECIDED) {
        if (lut.type == DLInstanceType::LUT) {
          impl.candidate_type = LutCandidateType::LUT;
        } else if (lut.type == DLInstanceType::SHIFT) {
          impl.candidate_type = LutCandidateType::SHIFT;
        } else if (lut.type == DLInstanceType::LRAM) {
          impl.candidate_type = LutCandidateType::LRAM;
        } else {
          openparfAssertMsg(false, "Error! A lut that is not LRAM SHIFT or LUT");
        }
      }
      return true;
    }
  }

  // Then check the odd slots
  // Here we need to perform the LUT compatibility checking
  for (IndexType i = 1; i < _param.CLB_capacity; i += _param.BLE_capacity) {
    if (impl.lut[i] == kIndexTypeMax && twoLUTsAreCompatible(_instArray[impl.lut[i - 1]], lut)) {
      impl.lut[i] = lut.id;
      return true;
    }
  }
  return false;
}

/// Try to fit a set of LUTs into a given candidate implementation using max matching algorithm
bool DLSolver::fitLUTsToCandidateImpl(const IndexVector &luts, Candidate::Implementation &impl) {
  // Build matching graph
  auto &mem       = _threadMemArray[omp_get_thread_num()];
  auto &graph     = *(mem.graphPtr);
  auto &nodes     = mem.nodes;
  auto &edges     = mem.edges;
  auto &edgePairs = mem.edgePairs;
  graph.clear();
  nodes.clear();
  edges.clear();
  edgePairs.clear();

  // Add nodes
  IndexType n = luts.size();
  for (IndexType i = 0; i < n; ++i) {
    nodes.emplace_back(graph.addNode());
  }
  // Add edges
  for (IndexType l = 0; l < n; ++l) {
    for (IndexType r = l + 1; r < n; ++r) {
      if (twoLUTsAreCompatible(_instArray[l], _instArray[r])) {
        edges.emplace_back(graph.addEdge(nodes[l], nodes[r]));
        edgePairs.emplace_back(l, r);
      }
    }
  }

  // Perform max matching
  lemon::MaxMatching<lemon::ListGraph> mm(graph);
  mm.run();
  if (n - mm.matchingSize() > _param.num_BLEs_per_CLB) {
    return false;
  }

  // Commit the legal solution
  // Unmatched LUTs
  IndexType idx = 0;
  for (IndexType i = 0; i < n; ++i) {
    if (mm.mate(nodes[i]) == lemon::INVALID) {
      // const auto &inst = _instArray[luts[i]];
      impl.lut[idx]     = luts[i];
      impl.lut[idx + 1] = kIndexTypeMax;
      idx += _param.BLE_capacity;
    }
  }
  // Matched LUTs
  for (IndexType i = 0; i < edges.size(); ++i) {
    if (mm.matching(edges[i])) {
      const auto &p     = edgePairs[i];
      impl.lut[idx]     = luts[p.first];
      impl.lut[idx + 1] = luts[p.second];
      idx += _param.BLE_capacity;
    }
  }
  // Fill the rest LUT slots with empty
  std::fill(impl.lut.begin() + idx, impl.lut.end(), kIndexTypeMax);
  return true;
}

/// Remove invalid candidates from the site
void DLSolver::removeInvalidCandidates(DLSite &site) {
  // Remove invalid candidates from the PQ
  auto &pq = site.next.pq;
  auto  it = pq.begin();
  while (it != pq.end()) {
    if (!candidateIsValid(*it)) {
      it = pq.erase(it);
    } else {
      ++it;
    }
  }

  // Remove invalid candidates from seed candidate list (scl)
  auto &scl  = site.curr.scl;
  auto  rmIt = std::remove_if(scl.begin(), scl.end(), [&](const Candidate &cand) { return !candidateIsValid(cand); });
  scl.erase(rmIt, scl.end());

  // If site.scl become empty, add site.det into it as the seed
  if (scl.empty()) {
    scl.emplace_back(site.det);
  }
}

/// Given a site, remove all its neighbor instances that have been commited
void DLSolver::removeCommittedNeighbors(DLSite &site) {
  for (IndexType &instId : site.nbr) {
    // An instance can be invalid if it has been commited to a site
    const auto &inst = _instArray[instId];
    if (inst.curr.detSiteId != kIndexTypeMax) {
      instId = kIndexTypeMax;
    }
  }
  site.nbr.erase(std::remove(site.nbr.begin(), site.nbr.end(), kIndexTypeMax), site.nbr.end());
}

/// Add more neighbor instances to the site if not enough
void DLSolver::addNeighbors(DLSite &site) {
  if (site.nbr.size() >= _param.minNeighbors || site.nbrGroupIdx >= site.numNbrGroups()) {
    // We do not add new neighbors if one of the following holds
    //   1) this site has enough neighbor instances
    //   2) there is no more neighbor instances within the maximum distance limit
    return;
  }

  // Add the next neighbor group into the active set
  IndexType beg = site.nbrRanges[site.nbrGroupIdx];
  IndexType end = site.nbrRanges[site.nbrGroupIdx + 1];
  for (IndexType i = beg; i < end; ++i) {
    // We do not add this instance if one of the following holds
    //   1) the instance has been commited to a site
    //   2) the instance is not compatible with the site.det
    const auto &inst = _instArray[site.nbrList[i]];
    if (inst.curr.detSiteId == kIndexTypeMax && addInstToCandidateIsFeasible(inst, site.det)) {
      site.nbr.push_back(site.nbrList[i]);
    }
  }
  ++site.nbrGroupIdx;

  // !!!!! Add the code below may worsen the QoR in experiments !!!!!
  //
  // Populate all candidates in site.next.pq into site.curr.scl
  // This is to make sure all newly added neighbors can consider all candidates in the PQ
  // Here we need to use next.pq since curr.pq may contain invalid candidates
  // site.curr.scl.assign(site.next.pq.begin(), site.next.pq.end());
}

/// Grow existing seed candidates in a site (scl) to generate new candidates
void DLSolver::createNewCandidates(DLSite &site) {
  // Try to generate new candidates by merging instances in site.nbr and seed candidates in site.scl
  for (const auto &seed : site.curr.scl) {
    for (IndexType instId : site.nbr) {
      // The candidate of (inst, seed) can be formed only if all of the following hold
      //   1) the seed does not contain the inst
      //   2) the site does not have candidate(inst, seed) in its PQ
      //   3) the candidate (inst, seed) is feasible
      Candidate   cand(seed);
      const auto &inst = _instArray[instId];
      if (addInstToSignature(inst, cand.sig) &&
          !site.next.pq.contain([&](const Candidate &c) { return c.sig == cand.sig; }) &&
          addInstToCandidateImpl(inst, cand)) {
        // Compute the candidate score
        // If the new candidate is pushed into the PQ successfully,
        // also save it in the scl to work as a seed candidate in the next DL iteration
        cand.score = computeCandidateScore(cand);
        if (site.next.pq.push(cand)) {
          site.next.scl.emplace_back(cand);
        }
      }
    }
  }

  // Remove candidates in site.next.scl that are not in the site.next.pq
  // This step is needed, because some new candidates can get into the PQ first,
  // then being kicked out by some other new candidates later
  if (!site.next.pq.empty()) {
    // Remove all candidates in scl that is worse than the worst candidate in the PQ
    Candidate::Comparator comp;
    auto                  it = std::remove_if(site.next.scl.begin(), site.next.scl.end(), [&](const Candidate &c) {
      return comp(site.next.pq.bottom(), c);
    });
    site.next.scl.erase(it, site.next.scl.end());
  }

  // Update stable iteration count
  if (!site.curr.pq.empty() && !site.next.pq.empty() && site.curr.pq.top() == site.next.pq.top()) {
    site.next.stable = site.curr.stable + 1;
  } else {
    site.next.stable = 0;
  }
}

/// Compute the score of a candidate
RealType DLSolver::computeCandidateScore(const Candidate &cand) {
  RealType    netShareScore = 0.0, wirelenImprov = 0.0;
  const auto &site = _siteMap[cand.siteId];

  // Collect all pins in this candidate then sort them from low to high
  // Note in this way, pins belonging to the same net will be adjacent
  auto       &pins = _threadMemArray[omp_get_thread_num()].idxVecA;
  pins.clear();
  for (IndexType instId : cand.sig) {
    const auto &arr = _instArray[instId].pinIdArray;
    pins.insert(pins.end(), arr.begin(), arr.end());
  }
  std::sort(pins.begin(), pins.end());

  if (pins.empty()) {
    return 0;
  }

  // We compute the total score in a net-by-net manner
  IndexType   maxNetDegree = std::max(_param.netShareScoreMaxNetDegree, _param.wirelenScoreMaxNetDegree);
  const auto *currNet      = &_netArray[_pinArray[pins[0]].netId];
  if (currNet->numPins() > maxNetDegree) {
    return 0;
  }

  IndexType numIntNets     = 0;
  IndexType numNets        = 0;
  auto     &currNetIntPins = _threadMemArray[omp_get_thread_num()].idxVecB;
  currNetIntPins.clear();
  currNetIntPins.push_back(pins[0]);
  for (IndexType i = 1; i < pins.size(); ++i) {
    IndexType netId = _pinArray[pins[i]].netId;
    if (netId == currNet->id) {
      // The current pin still share the same net with the previous pin
      currNetIntPins.push_back(pins[i]);
    } else {
      // The current pin does not share the same net with the previous pin any more
      // We here compute the net sharing and wirelength scores for the previous net (currNet)
      if (currNet->numPins() <= _param.netShareScoreMaxNetDegree) {
        ++numNets;
        numIntNets += (currNetIntPins.size() == currNet->numPins() ? 1 : 0);
        netShareScore += currNet->weight * (currNetIntPins.size() - 1.0) / (currNet->numPins() - 1.0);
      }
      if (currNet->numPins() <= _param.wirelenScoreMaxNetDegree) {
        wirelenImprov += computeWirelenImprov(*currNet, currNetIntPins, site.loc);
      }
      currNet = &_netArray[netId];
      if (currNet->numPins() > maxNetDegree) {
        break;
      }
      currNetIntPins.clear();
      currNetIntPins.push_back(pins[i]);
    }
  }
  // Need to handle the last net as well
  if (currNet->numPins() <= _param.netShareScoreMaxNetDegree) {
    ++numNets;
    numIntNets += (currNetIntPins.size() == currNet->numPins() ? 1 : 0);
    netShareScore += currNet->weight * (currNetIntPins.size() - 1.0) / (currNet->numPins() - 1.0);
  }
  if (currNet->numPins() <= _param.wirelenScoreMaxNetDegree) {
    wirelenImprov += computeWirelenImprov(*currNet, currNetIntPins, site.loc);
  }
  // Compute the final score and return
  netShareScore /= (1.0 + _param.extNetCountWt * (numNets - numIntNets));
  return netShareScore + _param.wirelenImprovWt * wirelenImprov;
}

/// Given a net and a set of pins in this net,
/// compute the wirelength improvment of moving the belonging instances of these pins to a given location
/// @param net   the given net, its bounding box (bbox) must be correct
/// @param pins  the set of pins in this net that will be moved
/// @param loc   the target moving location
/// The improvement can be a negative value, which means the wirelength gets worse
RealType DLSolver::computeWirelenImprov(const DLNet &net, const IndexVector &pins, const XY<RealType> &loc) const {
  if (net.numPins() == pins.size()) {
    // All the pins in this net are moved together
    Box<RealType> bbox(_pinArray[pins[0]].offset.x(),
                       _pinArray[pins[0]].offset.y(),
                       _pinArray[pins[0]].offset.x(),
                       _pinArray[pins[0]].offset.y());
    for (IndexType i = 1; i < pins.size(); ++i) {
      bbox.encompass(_pinArray[pins[i]].offset);
    }
    return net.weight * (_param.xWirelenWt * (net.bbox.width() - bbox.width()) +
                         _param.yWirelenWt * (net.bbox.height() - bbox.height()));
  }

  // The net bounding box after moving
  Box<RealType> bbox(net.bbox);

  // Update bbox.xl()
  if (loc.x() <= bbox.xl()) {
    bbox.setXL(loc.x());
  } else {
    // Find the first pin in net.pinIdArrayX that does not in 'pins'
    auto it = net.pinIdArrayX.begin();
    while (std::find(pins.begin(), pins.end(), *it) != pins.end()) {
      ++it;
    }
    const auto &pin  = _pinArray[*it];
    RealType    pinX = _instArray[pin.instId].x() + pin.offsetX();
    bbox.setXL(std::min(pinX, loc.x()));
  }

  // Update bbox.xh()
  if (loc.x() >= bbox.xh()) {
    bbox.setXH(loc.x());
  } else {
    // Find the last pin in net.pinIdArrayX that does not in 'pins'
    auto rit = net.pinIdArrayX.rbegin();
    while (std::find(pins.begin(), pins.end(), *rit) != pins.end()) {
      ++rit;
    }
    const auto &pin  = _pinArray[*rit];
    RealType    pinX = _instArray[pin.instId].x() + pin.offsetX();
    bbox.setXH(std::max(pinX, loc.x()));
  }

  // Update bbox.yl()
  if (loc.y() <= bbox.yl()) {
    bbox.setYL(loc.y());
  } else {
    // Find the first pin in net.pinIdArrayY that does not in 'pins'
    auto it = net.pinIdArrayY.begin();
    while (std::find(pins.begin(), pins.end(), *it) != pins.end()) {
      ++it;
    }
    const auto &pin  = _pinArray[*it];
    RealType    pinY = _instArray[pin.instId].y() + pin.offsetY();
    bbox.setYL(std::min(pinY, loc.y()));
  }

  // Update bbox.yh()
  if (loc.y() >= bbox.yh()) {
    bbox.setYH(loc.y());
  } else {
    // Find the last pin in net.pinIdArrayY that does not in 'pins'
    auto rit = net.pinIdArrayY.rbegin();
    while (std::find(pins.begin(), pins.end(), *rit) != pins.end()) {
      ++rit;
    }
    const auto &pin  = _pinArray[*rit];
    RealType    pinY = _instArray[pin.instId].y() + pin.offsetY();
    bbox.setYH(std::max(pinY, loc.y()));
  }
  return net.weight * (_param.xWirelenWt * (net.bbox.width() - bbox.width()) +
                       _param.yWirelenWt * (net.bbox.height() - bbox.height()));
}

/// Given a site, broadcast its top candidate to the constituting instances
void DLSolver::broadcastTopCandidate(const DLSite &site) {
  if (site.next.pq.empty()) {
    return;
  }
  RealType scoreImprov = site.next.pq.top().score - site.det.score;
  for (IndexType instId : site.next.pq.top().sig) {
    _instArray[instId].updateBestSite(site.id, scoreImprov);
  }
}

/// Perform synchronization between DL iterations
DLSolver::DLStatus DLSolver::runDLSync() {
  bool active  = false;
  bool illegal = false;

// Synchronize sites
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, _param.omp_dynamic_chunk_size)                    \
        reduction(|                                                                                                    \
                  : active)
  for (IndexType i = 0; i < numSites(); ++i) {
    auto &site = _siteMap[i];
    if (site.type != DLSiteType::DONTCARE) {
      site.curr = site.next;
      site.next.scl.clear();
      active |= (!site.curr.pq.empty() || !site.curr.scl.empty() || site.nbrGroupIdx < site.numNbrGroups());
    }
  }

// Synchronize instances
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, _param.omp_dynamic_chunk_size)                    \
        reduction(|                                                                                                    \
                  : illegal)
  for (IndexType i = 0; i < numInsts(); ++i) {
    auto &inst = _instArray[i];
    if (inst.type != DLInstanceType::DONTCARE && inst.curr.detSiteId == kIndexTypeMax) {
      inst.curr                 = inst.next;
      inst.next.bestSiteId      = kIndexTypeMax;
      inst.next.bestScoreImprov = kRealTypeMin;
      illegal |= (inst.curr.detSiteId == kIndexTypeMax);
    }
  }

  // Determine the DL status
  if (active) {
    return DLStatus::ACTIVE;
  }
  return illegal ? DLStatus::ILLEGAL : DLStatus::LEGAL;
}

/// Legalize illegal instances by ripup some legal instances
bool DLSolver::ripupLegalization() {
  // Collect instances that are not yet legal
  // We break all preclusters in this step
  IndexVector insts;
  IndexType   clk_ilg = 0;
  for (auto &inst : _instArray) {
    if (inst.type != DLInstanceType::DONTCARE) {
      inst.precluster.clear();
      inst.precluster.push_back(inst.id);
      if (inst.curr.detSiteId == kIndexTypeMax) {
        insts.push_back(inst.id);
      } else if (!instAndSiteClockCompatible(inst, _siteMap[inst.curr.detSiteId])) {
        insts.push_back(inst.id);
        auto &site = _siteMap[inst.curr.detSiteId];
        site.det.impl.reset();
        for (auto i : site.det.sig) {
          _instArray[i].curr.detSiteId = kIndexTypeMax;
          insts.push_back(i);
        }
        openparfAssert(site.det.siteId == site.id);
        site.det.siteId == kIndexTypeMax;
        inst.curr.detSiteId = kIndexTypeMax;
        clk_ilg++;
      }
    }
  }

  // Sort all instances by their area from high to low
  std::sort(insts.begin(), insts.end(), [&](IndexType a, IndexType b) {
    return _instArray[a].area > _instArray[b].area;
  });

  // count illegal instances
  IndexType count = 0;
  for (IndexType i : insts) {
    auto const &inst = _instArray[i];
    if (inst.curr.detSiteId == kIndexTypeMax) {
      ++count;
    }
  }
  openparfPrint(kInfo, "ripup %u (clock illegal %u) instances for legalization\n", count, clk_ilg);

  // Try to legalize illegal instances one by one
  for (IndexType i : insts) {
    // We first perform rip-up legalization
    // If fails, then perform greedy legalization

    // We need to perform this check even for the first iteration
    // because its binded nodes may legalize it already
    auto &inst = _instArray[i];
    if (inst.curr.detSiteId == kIndexTypeMax) {
      if (!ripupLegalizeInst(inst, _param.nbrDistEnd) && !greedyLegalizeInst(inst)) {
        return false;
      }
    }
  }
  return true;
}

/// Legalize an illegal instance by ripup other legal clusters
/// We require all instance displacement is less than the given limit
bool DLSolver::ripupLegalizeInst(DLInstance &inst, RealType maxD) {
  // Collect all legal site candidates within the given max distance
  std::vector<RipupCandidate> ripupCands(1, RipupCandidate(_num_LUTs, _num_FFs));

  IndexType                   xLo = std::max(inst.x() - maxD, (RealType) 0.0);
  IndexType                   yLo = std::max(inst.y() - maxD, (RealType) 0.0);
  IndexType                   xHi = std::min(inst.x() + maxD, _siteMap.xSize() - (RealType) 1.0);
  IndexType                   yHi = std::min(inst.y() + maxD, _siteMap.ySize() - (RealType) 1.0);
  for (IndexType x = xLo; x <= xHi; ++x) {
    for (IndexType y = yLo; y <= yHi; ++y) {
      const auto &site = _siteMap(x, y);
      if (instAndSiteAreCompatible(inst, site) && instAndSiteClockCompatible(inst, site)) {
        RealType dist = inst.loc.manhattanDistance(site.loc);
        if (dist < maxD && createRipupCandidate(inst, site, ripupCands.back())) {
          ripupCands.emplace_back(_num_LUTs, _num_FFs);
        }
      }
    }
  }
  ripupCands.pop_back();
  // Sort all rip-up candidates by their priority from high to low
  // Try to legalize the instance using the rip-up candidate one by one util a legal solution is found
  std::sort(ripupCands.begin(), ripupCands.end());
  for (const auto &ruc : ripupCands) {
    auto &site = _siteMap[ruc.siteId];
    if (ruc.legal) {
      site.det = ruc.cand;
      for (IndexType i : inst.precluster) {
        _instArray[i].curr.detSiteId = site.id;
      }
      return true;
    }
    if (ripupSiteAndLegalizeInstance(inst, site, maxD)) {
      return true;
    }
  }
  return false;
}

/// Create a ripup-and-legalize candidate for the given instance and site
bool DLSolver::createRipupCandidate(const DLInstance &inst, const DLSite &site, RipupCandidate &res) {
  res.siteId = site.id;
  res.cand   = site.det;
  res.legal  = addInstToCandidateImpl(inst, res.cand);
  if (res.legal) {
    // Adding the instance into the site is directly legal
    // The ripup score is the candidate score improvement
    addInstToSignature(inst, res.cand.sig);
    res.cand.score = computeCandidateScore(res.cand);
    res.score      = res.cand.score - site.det.score;
  } else {
    // Adding the instance into the site is not legal
    //
    // Compute the total instance area to be ripped up
    RealType area = 0.0;
    for (IndexType i : site.det.sig) {
      area += _instArray[i].area;
    }
    // Compute the wirelength improvement of puting the instance to the site
    RealType wirelenImprov = 0;
    for (IndexType pinId : inst.pinIdArray) {
      const auto &pin = _pinArray[pinId];
      const auto &net = _netArray[pin.netId];
      if (net.numPins() <= _param.wirelenScoreMaxNetDegree) {
        wirelenImprov += computeWirelenImprov(net, IndexVector{pinId}, site.loc);
      }
    }
    res.score = _param.wirelenImprovWt * wirelenImprov - site.det.score - area;
  }
  return true;
}

/// Rip up a site and legalize the given instance inthe the site
/// Legalize all the ripped instance using greedy approach
/// If cannot find a legal solution, then revert back and return false
bool DLSolver::ripupSiteAndLegalizeInstance(DLInstance &inst, DLSite &site, RealType maxD) {
  // Record the original status of all affected sites in case of rollback due to legalization failure
  std::vector<Candidate> dets;
  dets.reserve(site.det.sig.size());
  dets.emplace_back(site.det);

  // Rip up the site and put the instance into it
  for (IndexType i : site.det.sig) {
    _instArray[i].curr.detSiteId = kIndexTypeMax;
  }
  site.det.sig = inst.precluster;
  site.det.impl.reset();
  addInstToCandidateImpl(inst, site.det);
  site.det.score = computeCandidateScore(site.det);
  for (IndexType i : inst.precluster) {
    _instArray[i].curr.detSiteId = site.id;
  }

  // Find the best legal positions for each instance that was in the site
  // WARNING: Here we need to copy the signature vector, since we will push new elements into 'dets' which can
  // invalidate the range-based for loop
  IndexVector sig = dets[0].sig;
  for (IndexType ruInstId : sig) {
    const auto &ruInst = _instArray[ruInstId];
    if (ruInst.curr.detSiteId != kIndexTypeMax) {
      // One of this instance's preclustered instances has been legalized
      // So we don't need to legalize this instance again
      continue;
    }
    auto         beg = _spiralAccessor.begin();
    auto         end = _spiralAccessor.end(std::ceil(maxD + 1.0));
    XY<RealType> initXY(computeInstCentroid(ruInst.precluster));

    Candidate    bestCand(_num_LUTs, _num_FFs);
    RealType     bestScoreImprov = kRealTypeMin;
    for (auto it = beg; it != end; ++it) {
      XY<IntType> xy(IntType(initXY.x() + it->x()), IntType(initXY.y() + it->y()));
      if (!_bndBox.contain(xy)) {
        continue;
      }
      auto     &site = _siteMap(xy.x(), xy.y());
      Candidate cand(site.det);
      if (instAndSiteAreCompatible(ruInst, site) && instAndSiteClockCompatible(ruInst, site) &&
          addInstToCandidateImpl(ruInst, cand) && addInstToSignature(ruInst, cand.sig)) {
        // Adding the instance to the site is legal
        // If this is the first legal position found, set the expansion search radius
        if (bestScoreImprov == kRealTypeMin) {
          IntType r = _spiralAccessor.radius(it);
          r += _param.ripupExpansion;
          end = _spiralAccessor.end(std::min(_spiralAccessor.maxRadius(), r));
        }
        cand.score           = computeCandidateScore(cand);
        RealType scoreImprov = cand.score - site.det.score;
        if (scoreImprov > bestScoreImprov) {
          bestCand        = cand;
          bestScoreImprov = scoreImprov;
        }
      }
    }
    if (bestCand.siteId == kIndexTypeMax) {
      // Cannot find a legal position for this rip-up instance, so moving the instance to the site is illegal
      //
      // Revert all affected sites' clusters
      for (auto rit = dets.rbegin(); rit != dets.rend(); ++rit) {
        _siteMap[rit->siteId].det = *rit;
      }
      // Move all ripped instances back to their original sites
      for (IndexType i : site.det.sig) {
        _instArray[i].curr.detSiteId = site.id;
      }
      // Set the instance as illegal
      for (IndexType i : inst.precluster) {
        _instArray[i].curr.detSiteId = kIndexTypeMax;
      }
      return false;
    } else {
      // Record the original cluster at this site
      auto &st = _siteMap[bestCand.siteId];
      dets.emplace_back(st.det);
      // Move the ripped instances to this site
      st.det = bestCand;
      for (IndexType i : ruInst.precluster) {
        _instArray[i].curr.detSiteId = st.id;
      }
    }
  }
  // If the execution reaches here, all ripped instances found legal positions
  return true;
}

/// Greedily legalize all illegal instances after regular DL iterations
bool DLSolver::greedyLegalization() {
  // Collect instances that are not yet legal
  // We break all preclusters in this step
  IndexVector insts;
  for (auto &inst : _instArray) {
    if (inst.type != DLInstanceType::DONTCARE) {
      inst.precluster.clear();
      inst.precluster.push_back(inst.id);
      if (inst.curr.detSiteId == kIndexTypeMax) {
        insts.push_back(inst.id);
      }
    }
  }

  // Sort all instances by their area from high to low
  std::sort(insts.begin(), insts.end(), [&](IndexType a, IndexType b) {
    return _instArray[a].area > _instArray[b].area;
  });

  // Legalize illegal instances one by one
  for (IndexType i : insts) {
    if (!greedyLegalizeInst(_instArray[i])) {
      return false;
    }
  }
  return true;
}

/// Legalize an instance greedily
bool DLSolver::greedyLegalizeInst(DLInstance &inst) {
  auto         beg = _spiralAccessor.begin();
  auto         end = _spiralAccessor.end();
  XY<RealType> initXY(computeInstCentroid(inst.precluster));

  Candidate    bestCand(_num_LUTs, _num_FFs);
  RealType     bestScoreImprov = kRealTypeMin;
  for (auto it = beg; it != end; ++it) {
    XY<IntType> xy(IntType(initXY.x() + it->x()), IntType(initXY.y() + it->y()));
    if (!_bndBox.contain(xy)) {
      continue;
    }
    auto     &site = _siteMap(xy.x(), xy.y());
    Candidate cand(site.det);
    if (instAndSiteAreCompatible(inst, site) && instAndSiteClockCompatible(inst, site) &&
        addInstToCandidateImpl(inst, cand) && addInstToSignature(inst, cand.sig)) {
      // Adding the instance to the site is legal
      // If this is the first legal position found, set the expansion search radius
      if (bestScoreImprov == kRealTypeMin) {
        IntType r = _spiralAccessor.radius(it);
        r += _param.greedyExpansion;
        end = _spiralAccessor.end(std::min(_spiralAccessor.maxRadius(), r));
      }

      cand.score           = computeCandidateScore(cand);
      RealType scoreImprov = cand.score - site.det.score;
      if (scoreImprov > bestScoreImprov) {
        bestCand        = cand;
        bestScoreImprov = scoreImprov;
      }
    }
  }

  // Commit the found best legal solution
  if (bestCand.siteId != kIndexTypeMax) {
    _siteMap[bestCand.siteId].det = bestCand;
    for (IndexType i : inst.precluster) {
      _instArray[i].curr.detSiteId = bestCand.siteId;
    }
    return true;
  }
  return false;
}

/// Print the statistics for the DL solution
void DLSolver::printDLStatistics() const {
  if (_param.verbose < 1) {
    return;
  }

  /// Class to store statistics for a instance type
  struct InstanceTypeStats {
    IndexType             numInsts = 0;
    std::vector<RealType> movs;
  };
  // Collect instance type stats
  std::map<DLInstanceType, InstanceTypeStats> instStats;
  for (const auto &inst : _instArray) {
    auto &st = instStats[inst.type];
    ++st.numInsts;
    IndexType siteId = inst.curr.detSiteId;
    if (siteId != kIndexTypeMax) {
      st.movs.push_back(_siteMap[siteId].loc.manhattanDistance(inst.loc));
    }
  }
  // Process and print stats for each instance type
  IndexType maxD = std::ceil(_param.nbrDistEnd);
  for (auto &p : instStats) {
    std::stringstream ss;
    ss << std::setw(10) << toString(p.first) << ": ";
    auto &st = p.second;
    if (p.first != DLInstanceType::DONTCARE) {
      RealType avgMov, maxMov;
      if (st.movs.size() > 0) {
        avgMov = std::accumulate(st.movs.begin(), st.movs.end(), 0.0) / st.movs.size();
        std::sort(st.movs.begin(), st.movs.end());
        maxMov = st.movs.back();
      } else {
        avgMov = 0;
        maxMov = 0;
      }
      ss << "legal/total = " << st.movs.size() << "/" << st.numInsts << ", ";
      ss << "avg/max mov = " << std::fixed << std::setprecision(2) << avgMov << "/" << maxMov << ", ";
      ss << "freq [";
      auto curr = st.movs.begin();
      for (RealType d = 1.0; d <= maxD; d += 1.0) {
        auto next = std::lower_bound(curr, st.movs.end(), d);
        ss << std::setw(6) << next - curr;
        curr = next;
      }
      ss << std::setw(6) << st.movs.end() - curr;
      ss << "]";
    } else {
      ss << "total = " << st.numInsts;
    }
    openparfPrint(kInfo, "%s\n", ss.str().c_str());
  }

  /// Class to store statistics for a site type
  struct SiteTypeStats {
    IndexType total = 0;
    IndexType occup = 0;
    RealType  score = 0;
  };
  // Collect stats for each site type
  std::map<DLSiteType, SiteTypeStats> siteStats;
  for (const auto &site : _siteMap) {
    auto &st = siteStats[site.type];
    ++st.total;
    if (!site.det.sig.empty()) {
      ++st.occup;
      st.score += site.det.score;
    }
  }
  for (auto &p : siteStats) {
    std::stringstream ss;
    ss << std::setw(10) << toString(p.first) << ": ";
    auto &st = p.second;
    if (p.first != DLSiteType::DONTCARE) {
      ss << "occup/total = " << st.occup << "/" << st.total << ", score = " << st.score;
    } else {
      ss << "total = " << st.total;
    }
    openparfPrint(kInfo, "%s\n", ss.str().c_str());
  }
  auto wl = computeHPWL();
  openparfPrint(kInfo,
                "Post-DL HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                _param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y(),
                _param.xWirelenWt,
                wl.x(),
                _param.yWirelenWt,
                wl.y());
}

/// Print the statistics for the post legalization
void DLSolver::printPostLegalizationStatistics() const {
  if (_param.verbose < 1) {
    return;
  }
  auto wl = computeHPWL();
  openparfPrint(kInfo,
                "Post-Legalization HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                _param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y(),
                _param.xWirelenWt,
                wl.x(),
                _param.yWirelenWt,
                wl.y());
  for (auto &inst : _instArray) {
    if (inst.type != DLInstanceType::DONTCARE && !instAndSiteClockCompatible(inst, _siteMap[inst.curr.detSiteId])) {
      openparfPrint(kDebug,
                    "Clock illegal inst %i at %f %f\n",
                    inst.origId,
                    _siteMap[inst.curr.detSiteId].x(),
                    _siteMap[inst.curr.detSiteId].y());
    }
  }
}

/// Print current DL status for debugging
void DLSolver::printDLStatus() const {
  if (_param.verbose < 2) {
    return;
  }
  IndexType numLegal = 0;
  for (const auto &inst : _instArray) {
    if (inst.type == DLInstanceType::DONTCARE || inst.curr.detSiteId != kIndexTypeMax) {
      ++numLegal;
    }
  }
  auto hpwl = computeHPWL();
  openparfPrint(kDebug, "iter %u, #legal = %u / %u, HPWL = %.6E\n", _dlIter, numLegal, numInsts(), hpwl.x() + hpwl.y());
}

/// Compute HPWL of current solution
/// For instances that has been assigned to sites, we use their new locations
/// For instances that has not been assigned, we use thier initial locations
XY<RealType> DLSolver::computeHPWL() const {
  XY<RealType>  res(0, 0);
  Box<RealType> bbox;
  for (const auto &net : _netArray) {
    bbox.set(kRealTypeMax, kRealTypeMax, kRealTypeMin, kRealTypeMin);
    for (IndexType pinId : net.pinIdArray) {
      const auto &pin  = _pinArray[pinId];
      const auto &inst = _instArray[pin.instId];
      if (inst.curr.detSiteId == kIndexTypeMax) {
        bbox.encompass(inst.loc + pin.offset);
      } else {
        bbox.encompass(_siteMap[inst.curr.detSiteId].loc + pin.offset);
      }
    }
    res.set(res.x() + net.weight * bbox.width(), res.y() + net.weight * bbox.height());
  }
  return res;
}

/// Top function to run slot assignment
void        DLSolver::runSlotAssign() {
       // Perform slot assignment
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, _param.omp_dynamic_chunk_size)
  for (IndexType siteId = 0; siteId < numSites(); ++siteId) {
    auto &site = _siteMap[siteId];
    if (site.det.sig.empty()) {
      continue;
    }
    auto &mem = _threadMemArray[omp_get_thread_num()].slotAssignMem;
    initSlotAssign(site.det, mem);
    computeLUTScoreAndScoreImprov(mem);
    pairLUTs(mem);
    assignLUTsAndFFs(mem);
    orderBLEs(mem);
  }
  openparfPrint(kInfo, "Slot assignment done\n");
}

/// Initialize the slot assignment memory
void DLSolver::initSlotAssign(Candidate &cand, SlotAssignMemory &mem) {
  mem.candPtr = &cand;
  auto &impl  = cand.impl;

  // Collect all LUTs in this candidate implementation
  mem.lut.clear();
  for (IndexType id : impl.lut) {
    if (id != kIndexTypeMax) {
      mem.lut.push_back(id);
    }
  }

  // Collect all FFs
  mem.ff.clear();
  for (IndexType id : impl.ff) {
    if (id != kIndexTypeMax) {
      mem.ff.push_back(id);
    }
  }

  // Pre-assign control sets
  // Note that the original impl FF assignment is feasible but it is optimized for minimum resource usage
  // Therefore, to enlarge the solution space exploration, we do following modifications based on the original control
  // set assignment:
  //     (1) if ce[1] (ce[3]) is empty, we set it to ce[0] (ce[2]),
  //     (2) if (cksr[1], ce[2], ce[3]) are empty, we set them to (cksr[0], ce[0], ce[1])
  mem.cksr  = impl.cksr;
  mem.ce    = impl.ce;
  mem.ce[1] = (mem.ce[1] == kIndexTypeMax ? mem.ce[0] : mem.ce[1]);
  mem.ce[3] = (mem.ce[3] == kIndexTypeMax ? mem.ce[2] : mem.ce[3]);
  if (mem.cksr[1] == kIndexTypeMax) {
    mem.cksr[1] = mem.cksr[0];
    mem.ce[2]   = mem.ce[0];
    mem.ce[3]   = mem.ce[1];
  }
}

/// Compute all single-LUT and paired-LUT scores and score improvement
void DLSolver::computeLUTScoreAndScoreImprov(SlotAssignMemory &mem) {
  // Compute the best single-LUT BLE for each LUT
  mem.bleS.resize(mem.lut.size());
  for (IndexType i = 0; i < mem.lut.size(); ++i) {
    auto &ble  = mem.bleS[i];
    ble.lut[0] = mem.lut[i];
    ble.lut[1] = kIndexTypeMax;
    findBestFFs(mem.ff, mem.cksr, mem.ce, ble);
  }

  // Collect all feasible LUT pairs and compute their best scores and score improvement
  mem.bleP.clear();
  for (IndexType aIdx = 0; aIdx < mem.lut.size(); ++aIdx) {
    const auto &lutA = _instArray[mem.lut[aIdx]];
    for (IndexType bIdx = aIdx + 1; bIdx < mem.lut.size(); ++bIdx) {
      const auto &lutB = _instArray[mem.lut[bIdx]];
      if (!twoLUTsAreCompatible(lutA, lutB)) {
        continue;
      }
      mem.bleP.emplace_back();
      auto &ble  = mem.bleP.back();
      ble.lut[0] = mem.lut[aIdx];
      ble.lut[1] = mem.lut[bIdx];
      findBestFFs(mem.ff, mem.cksr, mem.ce, ble);

      // We define the score improvement of a compatible LUT pair (a, b) as
      //     improv(a, b) = max(BLEScore(a, b, *, *)) - max(BLEScore(a, -, *, *)) - max(BLEScore(b, -, *, *))
      ble.improv = ble.score - mem.bleS[aIdx].score - mem.bleS[bIdx].score;
    }
  }
}

/// Given LUTs in a BLE and two control sets, find the best FF(s)
void DLSolver::findBestFFs(const IndexVector              &ff,
                           const std::array<IndexType, 2> &cksr,
                           const std::array<IndexType, 4> &ce,
                           SlotAssignMemory::BLE          &ble) {
  SlotAssignMemory::BLE tempBLE(ble);
  findBestFFs(ff, cksr[0], ce[0], ce[1], ble);
  findBestFFs(ff, cksr[1], ce[2], ce[3], tempBLE);
  if (tempBLE.score > ble.score) {
    ble = tempBLE;
  }
}

/// Given LUTs in a BLE and a control set, find the best FF(s)
void DLSolver::findBestFFs(const IndexVector     &ff,
                           IndexType              cksr,
                           IndexType              ce0,
                           IndexType              ce1,
                           SlotAssignMemory::BLE &ble) {
  // Find the single FF or FF pairs that gives the best score
  ble.score = 0;
  ble.ff[0] = ble.ff[1] = kIndexTypeMax;
  for (IndexType aIdx = 0; aIdx < ff.size(); ++aIdx) {
    const auto &ffA = _instArray[ff[aIdx]];
    // Single FF
    if (ffA.cksr == cksr && (ffA.ce == ce0 || ffA.ce == ce1)) {
      RealType score = computeBLEScore(ble.lut[0], ble.lut[1], ff[aIdx], kIndexTypeMax);
      if (score > ble.score) {
        ble.ff[0] = ff[aIdx];
        ble.ff[1] = kIndexTypeMax;
        if (ffA.ce != ce0) {
          std::swap(ble.ff[0], ble.ff[1]);
        }
        ble.score = score;
      }
    }
    // FF pairs
    for (IndexType bIdx = aIdx + 1; bIdx < ff.size(); ++bIdx) {
      const auto &ffB = _instArray[ff[bIdx]];
      if (ffA.cksr == cksr && ffB.cksr == cksr &&
          ((ffA.ce == ce0 && ffB.ce == ce1) || (ffA.ce == ce1 && ffB.ce == ce0))) {
        RealType score = computeBLEScore(ble.lut[0], ble.lut[1], ff[aIdx], ff[bIdx]);
        if (score > ble.score) {
          ble.ff[0] = ff[aIdx];
          ble.ff[1] = ff[bIdx];
          if (ffA.ce != ce0) {
            std::swap(ble.ff[0], ble.ff[1]);
          }
          ble.score = score;
        }
      }
    }
  }
}

/// Perform max-weighted matching based LUT pairing
void DLSolver::pairLUTs(SlotAssignMemory &mem) {
  auto &graph = *(mem.graphPtr);
  auto &nodes = mem.nodes;
  auto &edges = mem.edges;
  graph.clear();
  nodes.clear();
  edges.clear();
  lemon::ListGraph::EdgeMap<FlowIntType> wtMap(graph);

  // Get LUT ID to index mapping
  auto                                  &idxMap = mem.idxMap;
  idxMap.clear();
  for (IndexType i = 0; i < mem.lut.size(); ++i) {
    idxMap[mem.lut[i]] = i;
  }

  // Build the graph use LUT pair score improvement as the edge weights
  for (IndexType i = 0; i < mem.lut.size(); ++i) {
    nodes.emplace_back(graph.addNode());
  }
  for (const auto &ble : mem.bleP) {
    edges.emplace_back(graph.addEdge(nodes[idxMap[ble.lut[0]]], nodes[idxMap[ble.lut[1]]]));
    wtMap[edges.back()] = ble.improv * _param.slotAssignFlowWeightScale;
  }

  // Use iterative max-weighted matching to find the best legal LUT pairing
  while (true) {
    lemon::MaxWeightedMatching<lemon::ListGraph, lemon::ListGraph::EdgeMap<FlowIntType>> mwm(graph, wtMap);
    mwm.run();
    if (nodes.size() - mwm.matchingSize() <= _param.num_BLEs_per_CLB) {
      // Collect the LUT pairing solution
      mem.ble.clear();
      for (IndexType i = 0; i < edges.size(); ++i) {
        if (mwm.matching(edges[i])) {
          mem.ble.emplace_back(mem.bleP[i]);
        }
      }
      for (IndexType i = 0; i < nodes.size(); ++i) {
        if (mwm.mate(nodes[i]) == lemon::INVALID) {
          mem.ble.emplace_back(mem.bleS[i]);
        }
      }
      return;
    }
    // Increase all edge weight to get a tighter LUT pairing solution
    FlowIntType incr = _param.slotAssignFlowWeightIncr * _param.slotAssignFlowWeightScale;
    for (const auto &e : edges) {
      wtMap[e] += incr;
    }
  }
}

/// Perform actual LUT and FF assignment after the LUT pairing
void DLSolver::assignLUTsAndFFs(SlotAssignMemory &mem) {
  // Reset the existing slot assignment
  auto &impl = mem.candPtr->impl;
  std::fill(impl.lut.begin(), impl.lut.end(), kIndexTypeMax);
  std::fill(impl.ff.begin(), impl.ff.end(), kIndexTypeMax);
  std::fill(impl.cksr.begin(), impl.cksr.end(), kIndexTypeMax);
  std::fill(impl.ce.begin(), impl.ce.end(), kIndexTypeMax);

  // Reset the score of each BLE slot
  auto &scores = mem.scores;
  scores.assign(_param.num_BLEs_per_CLB, 0.0);

  // Sort legal LUT/LUT pairs by their score from high to low
  std::sort(mem.ble.begin(), mem.ble.end(), [&](const SlotAssignMemory::BLE &a, const SlotAssignMemory::BLE &b) {
    return a.score > b.score;
  });

  // Record the number of available LUT pair slots in low/high half of the slice
  IndexType availLo = _param.num_BLEs_per_half_CLB;
  IndexType availHi = _param.num_BLEs_per_half_CLB;

  // Assign LUTs one by one and determine thier best FFs at the same time
  for (const auto &estBLE : mem.ble) {
    // Get the best BLE in low/high half of the slice
    SlotAssignMemory::BLE bleLo(estBLE);
    SlotAssignMemory::BLE bleHi(estBLE);
    findBestFFs(mem.ff, mem.cksr[0], mem.ce[0], mem.ce[1], bleLo);
    findBestFFs(mem.ff, mem.cksr[1], mem.ce[2], mem.ce[3], bleHi);

    // Try to fit the found BLE in the preferred feasible half slice
    IndexType   lh  = ((availLo && bleLo.score > bleHi.score) || !availHi ? 0 : 1);
    const auto &ble = (lh ? bleHi : bleLo);
    (lh ? availHi : availLo) -= 1;

    IndexType beg = lh * _param.half_CLB_capacity;
    IndexType end = beg + _param.half_CLB_capacity;
    for (IndexType idx = beg; idx < end; idx += _param.BLE_capacity) {
      IndexType pos = idx % _param.CLB_capacity;
      if (impl.lut[pos] == kIndexTypeMax && impl.lut[pos + 1] == kIndexTypeMax) {
        // Realize LUT assignment
        // Assign LUTs with more inputs at odd slots
        // In this way we can also make sure that all LUT6 are assigned at odd positions
        IndexType demA    = (ble.lut[0] == kIndexTypeMax ? 0 : _instArray[ble.lut[0]].dem);
        IndexType demB    = (ble.lut[1] == kIndexTypeMax ? 0 : _instArray[ble.lut[1]].dem);
        IndexType flip    = (demA > demB ? 1 : 0);
        impl.lut[pos]     = ble.lut[flip];
        impl.lut[pos + 1] = ble.lut[1 - flip];

        // Realize FF assignment
        for (IndexType k : {0, 1}) {
          impl.ff[pos + k] = ble.ff[k];
          if (ble.ff[k] != kIndexTypeMax) {
            const auto &ff      = _instArray[ble.ff[k]];
            impl.cksr[lh]       = ff.cksr;
            impl.ce[2 * lh + k] = ff.ce;

            // Remove the FF assigned from the active list
            mem.ff.erase(std::find(mem.ff.begin(), mem.ff.end(), ble.ff[k]));
          }
        }
        scores[pos / _param.BLE_capacity] = ble.score;
        break;
      }
    }
  }

  // Assign the rest of unassigned FFs
  //
  // We iteratively add one FF at a time
  // Each time, we add the FF gives the best score improvement
  while (!mem.ff.empty()) {
    // Find the FF gives the best score improvement
    IndexType bestFFIdx = kIndexTypeMax, bestPos = kIndexTypeMax;
    RealType  bestImprov = kRealTypeMin;
    for (IndexType ffIdx = 0; ffIdx < mem.ff.size(); ++ffIdx) {
      const auto &ff = _instArray[mem.ff[ffIdx]];
      for (IndexType pos = 0; pos < _param.CLB_capacity; ++pos) {
        IndexType lh = pos / _param.half_CLB_capacity;
        IndexType oe = pos % _param.BLE_capacity;
        if (impl.ff[pos] == kIndexTypeMax && ff.cksr == mem.cksr[lh] && ff.ce == mem.ce[2 * lh + oe]) {
          IndexType k     = pos / _param.BLE_capacity * _param.BLE_capacity;
          impl.ff[pos]    = ff.id;
          RealType improv = computeBLEScore(impl.lut[k], impl.lut[k + 1], impl.ff[k], impl.ff[k + 1]) -
                            scores[k / _param.BLE_capacity];
          impl.ff[pos] = kIndexTypeMax;
          if (improv > bestImprov) {
            bestFFIdx  = ffIdx;
            bestPos    = pos;
            bestImprov = improv;
          }
        }
      }
    }
    // Realize the best FF assignment found
    const auto &bestFF   = _instArray[mem.ff[bestFFIdx]];
    IndexType   lh       = bestPos / _param.half_CLB_capacity;
    IndexType   oe       = bestPos % _param.BLE_capacity;
    impl.ff[bestPos]     = bestFF.id;
    impl.cksr[lh]        = bestFF.cksr;
    impl.ce[2 * lh + oe] = bestFF.ce;

    // Remove the best FF found from the active list
    mem.ff.erase(mem.ff.begin() + bestFFIdx);

    // Update the BLE slot score
    scores[bestPos / _param.BLE_capacity] += bestImprov;
  }
}

/// Reorder BLEs in each half slice to reverse their original ordering in the given placement
void DLSolver::orderBLEs(SlotAssignMemory &mem) {
  // Sort BLEs in each half slice by their Y centroid coodiantes (cen.y - site.y)
  auto    &impl  = mem.candPtr->impl;
  RealType siteY = _siteMap[mem.candPtr->siteId].y();
  for (IndexType lh : {0, 1}) {
    mem.ble.clear();
    IndexType beg = lh * _param.half_CLB_capacity;
    IndexType end = beg + _param.half_CLB_capacity;
    for (IndexType offset = beg; offset < end; offset += _param.BLE_capacity) {
      mem.ble.emplace_back();
      auto &ble   = mem.ble.back();
      auto &insts = mem.idxVec;
      insts.clear();
      for (IndexType k : {0, 1}) {
        if (impl.lut[offset + k] != kIndexTypeMax) {
          insts.push_back(impl.lut[offset + k]);
          ble.lut[k] = impl.lut[offset + k];
        }
        if (impl.ff[offset + k] != kIndexTypeMax) {
          insts.push_back(impl.ff[offset + k]);
          ble.ff[k] = impl.ff[offset + k];
        }
      }
      // We use ble.score to store centroid.y - site.y of this BLE
      ble.score = (insts.empty() ? 0 : computeInstCentroid(insts).y() - siteY);
    }
    // Sort BLEs in this half slice by their centroid.y - site.y from low to high
    // Put them back to the implementation
    std::sort(mem.ble.begin(), mem.ble.end(), [&](const SlotAssignMemory::BLE &a, const SlotAssignMemory::BLE &b) {
      return a.score < b.score;
    });
    for (IndexType i = 0; i < mem.ble.size(); ++i) {
      const auto &ble    = mem.ble[i];
      IndexType   offset = lh * _param.half_CLB_capacity + i * _param.BLE_capacity;
      for (IndexType k : {0, 1}) {
        impl.lut[offset + k] = ble.lut[k];
        impl.ff[offset + k]  = ble.ff[k];
      }
    }
  }
}

/// Cache the DL solution back to each instance
void DLSolver::cacheSolution() {
  for (const auto &site : _siteMap) {
    for (IndexType z = 0; z < _param.CLB_capacity; ++z) {
      for (IndexType id : {site.det.impl.lut[z], site.det.impl.ff[z]}) {
        if (id != kIndexTypeMax) {
          _instArray[id].sol.set(site.id, z);
        }
      }
    }
  }
  openparfPrint(kDebug, "cacheSolution done.\n");
}

}   // namespace direct_lg
OPENPARF_END_NAMESPACE
