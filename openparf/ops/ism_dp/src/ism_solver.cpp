/**
 * File              : ism_solver.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.02.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "ops/ism_dp/src/ism_solver.h"

// C system headers
#include <omp.h>

// C++ standard library headers
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {
/// Constructor
ISMSolver::ISMSolver(const ISMProblem &prob, const ISMParam &param, IndexType num_threads, bool honorClockConstraints)
    : _param(param),
      _spiralAccessor(std::max(prob.siteXYs.xSize(), prob.siteXYs.ySize())),
      _bndBox(0, 0, prob.siteXYs.xSize() - 1, prob.siteXYs.ySize() - 1),
      _num_threads(num_threads),
      _honorClockConstraints(honorClockConstraints) {}

/// Build the ISM problem
void ISMSolver::buildISMProblem(const ISMProblem &prob) {
  // Build netlist and site map
  buildNetlist(prob);
  buildSiteMap(prob);

  // Sort netlist and remove netlist redundancy
  sortNetlist();
  removeNetlistRedundancy();
}

/// Build the netlist of the ISM problem
void ISMSolver::buildNetlist(const ISMProblem &prob) {
  // Construct instances and set their type, ck/sr, ce, site ID, and mate ID
  _instArray.clear();
  _instArray.resize(prob.instTypes.size());
  for (IndexType i = 0; i < _instArray.size(); ++i) {
    auto &inst      = _instArray[i];
    inst.origId     = i;
    inst.id         = i;
    inst.type       = prob.instTypes[i];
    inst.cksr       = prob.instCKSR[i];
    inst.ce         = prob.instCE[i];
    inst.mates      = prob.instToMates[i];
    inst.sol.siteId = prob.instToSite[i];
    inst.sol.ceFlip = false;
    if (inst.sol.isFixed()) {
      inst.sol.fixedSol = prob.instXYs[i];
    }
    if (_honorClockConstraints && i < prob.instCKs.size()) {
      inst.ckArray = prob.instCKs[i];
    }
  }
  // Build clock constraint related data structures
  if (_honorClockConstraints) _isCKAllowedInSite = prob.isCKAllowedInSite;

  // Construct nets and set their weights
  _netArray.clear();
  _netArray.resize(prob.netWts.size());
  for (IndexType i = 0; i < _netArray.size(); ++i) {
    auto &net  = _netArray[i];
    net.weight = prob.netWts[i];
  }

  // Construct pins and set their offsets, instance IDs and net IDs
  // We also need to add pin IDs into their corresponding instances and nets
  _pinArray.clear();
  _pinArray.resize(prob.pinOffsets.size());
  for (IndexType i = 0; i < _pinArray.size(); ++i) {
    auto &pin  = _pinArray[i];
    pin.offset = prob.pinOffsets[i];
    pin.instId = prob.pinToInst[i];
    pin.netId  = prob.pinToNet[i];
    _instArray[pin.instId].pinIdArray.push_back(i);
    _netArray[pin.netId].pinIdArray.push_back(i);
  }
}

/// Build the site map
void ISMSolver::buildSiteMap(const ISMProblem &prob) {
  // Set instance ID in each site
  _siteMap.clear();
  _siteMap.resize(prob.siteXYs.sizes());
  for (IndexType i = 0; i < _instArray.size(); ++i) {
    if (prob.instToSite[i] != kIndexTypeMax) {
      _siteMap[prob.instToSite[i]].instId = i;
    }
  }

  // Set site locations and control sets
  _ctrlSetArray.assign(_siteMap.size() / _param.numSitePerCtrlSet, ISMCtrlSet());
  for (IndexType i = 0; i < _siteMap.size(); ++i) {
    auto &site     = _siteMap[i];
    site.loc       = prob.siteXYs[i];
    site.ctrlSetId = prob.siteToCtrlSet[i];
    site.mateId    = prob.siteToMate[i];
    if (site.ctrlSetId != kIndexTypeMax) {
      auto &cs = _ctrlSetArray[site.ctrlSetId];
      if (site.instId != kIndexTypeMax) {
        const auto &inst = _instArray[site.instId];
        if (inst.cksr != kIndexTypeMax) {
          cs.cksr = inst.cksr;
          ++cs.cksrCount;
        }
        for (IndexType k : {0, 1}) {
          if (inst.ce[k] != kIndexTypeMax) {
            cs.ce[k] = inst.ce[k];
            ++cs.ceCount[k];
          }
        }
      }
    }
  }
}

/// Sort instance, net, and pin objects in the netlist
/// The sorted netlist should have the following properties
///   1) nets are sorted by their number of pins from low to high
///   2) instances without pins are after instances with pins
///   4) pins are sorted by their net IDs from low to high, for pins with the same net IDs, further sort them by their
///   instance IDs
void ISMSolver::sortNetlist() {
  // Sort nets
  auto netCmp = [&](IndexType a, IndexType b) { return _netArray[a].numPins() < _netArray[b].numPins(); };
  sortNets(netCmp);

  // Sort instances
  auto instCmp = [&](IndexType a, IndexType b) { return _instArray[a].numPins() > _instArray[b].numPins(); };
  sortInstances(instCmp);
  // Get the number of instances that have pins
  _numPhyInsts =
          std::find_if(_instArray.begin(), _instArray.end(), [&](const ISMInstance &i) { return i.numPins() == 0; }) -
          _instArray.begin();

  // Sort pins
  auto pinCmp = [&](IndexType a, IndexType b) {
    const auto &pinA = _pinArray[a];
    const auto &pinB = _pinArray[b];
    return (pinA.netId == pinB.netId ? pinA.instId < pinB.instId : pinA.netId < pinB.netId);
  };
  sortPins(pinCmp);
}

/// Sort nets using the given comparator
void ISMSolver::sortNets(std::function<bool(IndexType, IndexType)> cmp) {
  // Get net ordering and ranking
  IndexVector netOrder(numNets());
  IndexVector netRank(numNets());
  std::iota(netOrder.begin(), netOrder.end(), 0);
  std::sort(netOrder.begin(), netOrder.end(), cmp);
  for (IndexType rank = 0; rank < numNets(); ++rank) {
    netRank[netOrder[rank]] = rank;
  }

  // Construct the new netArray
  std::vector<ISMNet> netArray(numNets());
  for (IndexType id = 0; id < numNets(); ++id) {
    netArray[netRank[id]] = _netArray[id];
  }
  _netArray = std::move(netArray);

  // Update netId in pins
  for (auto &pin : _pinArray) {
    pin.netId = netRank[pin.netId];
  }
}

/// Sort pins using the given comparator
void ISMSolver::sortPins(std::function<bool(IndexType, IndexType)> cmp) {
  // Get pin ordering and ranking
  IndexVector pinOrder(numPins());
  IndexVector pinRank(numPins());
  std::iota(pinOrder.begin(), pinOrder.end(), 0);
  std::sort(pinOrder.begin(), pinOrder.end(), cmp);
  for (IndexType rank = 0; rank < numPins(); ++rank) {
    pinRank[pinOrder[rank]] = rank;
  }

  // Construct the new pinArray
  std::vector<ISMPin> pinArray(numPins());
  for (IndexType id = 0; id < numPins(); ++id) {
    pinArray[pinRank[id]] = _pinArray[id];
  }
  _pinArray = std::move(pinArray);

  // Update pinId in nets and instances
  for (auto &net : _netArray) {
    for (IndexType &pinId : net.pinIdArray) {
      pinId = pinRank[pinId];
    }
  }
  for (auto &inst : _instArray) {
    for (IndexType &pinId : inst.pinIdArray) {
      pinId = pinRank[pinId];
    }
  }
}

/// Sort instances using the given comparator
void ISMSolver::sortInstances(std::function<bool(IndexType, IndexType)> cmp) {
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
  std::vector<ISMInstance> instArray(numInsts());
  for (IndexType id = 0; id < numInsts(); ++id) {
    instArray[instRank[id]] = _instArray[id];
  }
  _instArray = std::move(instArray);

  // Update instId in instances
  for (auto &inst : _instArray) {
    inst.id = instRank[inst.id];
    for (IndexType &m : inst.mates) {
      m = instRank[m];
    }
  }
  // Update instId in pins
  for (auto &pin : _pinArray) {
    pin.instId = instRank[pin.instId];
  }
  // Update instId in site map
  for (auto &site : _siteMap) {
    site.instId = (site.instId == kIndexTypeMax ? kIndexTypeMax : instRank[site.instId]);
  }
}

/// Remove redundancy in netlist. This function must be called after function sortNetlist()
///   1) for each group of pins that are in the same net and also in the same instance, we only keep one of them
///   2) remove all 0- and 1-pin nets and their pins
void ISMSolver::removeNetlistRedundancy() {
  removeRedundantPins();
  removeRedundantNets();
}

/// Pins with the same belonging net and instance are redundant
/// For each group of redundant pins, we only keep one of them in the netlist and remove the other
/// Note that this function assumes the netlist is sorted
void ISMSolver::removeRedundantPins() {
  if (_pinArray.empty()) {
    return;
  }

  // Pins are sorted by net IDs and then instance IDs
  // So if two pins have the same net ID and instance ID, they must be adjacent

  // 'pinIdMapping' is the old pin ID to new pin ID mapping
  // 'newPinArray' is the new pinArray
  IndexVector         pinIdMapping(_pinArray.size(), kIndexTypeMax);
  std::vector<ISMPin> newPinArray;
  newPinArray.reserve(_pinArray.size());

  // It is safe to set the first pin as valid
  pinIdMapping[0] = 0;
  newPinArray.emplace_back(_pinArray.front());

  // Start the redundancy check
  auto      beg          = _pinArray.begin();
  auto      end          = _pinArray.end();
  auto      prev         = beg;
  auto      curr         = beg + 1;
  IndexType numValidPins = 1;
  while (curr != end) {
    if (curr->netId == prev->netId && curr->instId == prev->instId) {
      // The pin pointed by 'curr' is redundant
      ++curr;
    } else {
      // The pin pointed by 'curr' is no redundant
      pinIdMapping[curr - beg] = numValidPins++;
      newPinArray.emplace_back(*curr);
      prev = curr++;
    }
  }
  // Set the new pin array and update the pin IDs in instances and nets
  _pinArray = std::move(newPinArray);
  applyPinIdMapping(pinIdMapping);
}

inline XY<RealType> ISMSolver::getLoc(IndexType inst_id) const {
  auto &    sol     = _instArray[inst_id].sol;
  IndexType site_id = sol.siteId;
  if (sol.isFixed()) {
    return sol.fixedSol;
  } else {
    return _siteMap[site_id].loc;
  }
}

/// Remove all 0- and 1-pin nets and their pins
void ISMSolver::removeRedundantNets() {
  IndexType   numNets = 0;
  IndexVector netIdMapping(_netArray.size(), kIndexTypeMax);
  for (IndexType i = 0; i < _netArray.size(); ++i) {
    if (_netArray[i].numPins() > 1) {
      _netArray[numNets] = _netArray[i];
      netIdMapping[i]    = numNets++;
    }
  }
  _netArray.resize(numNets);
  // Update net IDs in pins
  for (auto &pin : _pinArray) {
    pin.netId = netIdMapping[pin.netId];
  }

  // For 1-pin net, we also need to remove their single pins
  IndexType   numPins = 0;
  IndexVector pinIdMapping(_pinArray.size(), kIndexTypeMax);
  for (IndexType i = 0; i < _pinArray.size(); ++i) {
    if (_pinArray[i].netId != kIndexTypeMax) {
      _pinArray[numPins] = _pinArray[i];
      pinIdMapping[i]    = numPins++;
    }
  }
  _pinArray.resize(numPins);
  // Update the pin IDs in instances and nets
  applyPinIdMapping(pinIdMapping);
}

/// Given a old pin ID to new pin ID mapping, update all pin IDs in instances and nets
void ISMSolver::applyPinIdMapping(const IndexVector &pinIdMapping) {
  // Update pin IDs in instances
  for (auto &inst : _instArray) {
    for (IndexType &pinId : inst.pinIdArray) {
      pinId = pinIdMapping[pinId];
    }
    inst.pinIdArray.erase(std::remove(inst.pinIdArray.begin(), inst.pinIdArray.end(), kIndexTypeMax),
                          inst.pinIdArray.end());
  }
  // Update pin IDs in nets
  for (auto &net : _netArray) {
    for (IndexType &pinId : net.pinIdArray) {
      pinId = pinIdMapping[pinId];
    }
    net.pinIdArray.erase(std::remove(net.pinIdArray.begin(), net.pinIdArray.end(), kIndexTypeMax),
                         net.pinIdArray.end());
  }
}

/// Top function to run the ISM
void ISMSolver::run() {
  initISM();

  // Iterativly run ISM until the stop condition holds
  std::vector<std::vector<IndexType>> indepSets;
  while (!stopCondition()) {
    // Build a set of independent independent sets that can be placed independently
    buildIndependentIndepSets(_indepSet, _sets);

// For each of these sets, perform ISM
#pragma omp parallel for num_threads(_num_threads) schedule(dynamic, OMP_DYNAMIC_CHUNK_SIZE)
    for (IndexType i = 0; i < _sets.size(); ++i) {
      const auto &set = _sets[i];
      auto &      mem = _threadMemArray[omp_get_thread_num()];
      computeNetBBoxes(set, mem);
      computeCostMatrix(set, mem);
      computeMatching(mem);
      realizeMatching(set, mem);
    }
  }
  if (_param.verbose > 0) {
    computeAllNetBBoxes();
    auto wl = computeHPWL();
    openparfPrint(kInfo,
                  "Post-ISM HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                  _param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y(),
                  _param.xWirelenWt,
                  wl.x(),
                  _param.yWirelenWt,
                  wl.y());
  }
}

/// Perform ISM initialization
void ISMSolver::initISM() {
  // Allocate memory
  _indepSet.dep.resize(numInsts());
  _threadMemArray.resize(omp_get_max_threads());

  // Initialize _priority
  // _priority contains all instances can be ISM seeds,
  // which need to have pins (non-fillers) and valid instance types (non-fixed)
  _priority.reserve(_numPhyInsts);
  for (IndexType i = 0; i < _numPhyInsts; ++i) {
    const auto &inst = _instArray[i];
    if (inst.type != kIndexTypeMax) {
      _priority.push_back(i);
    }
  }

  // Initialize connected instances in each instance
  initInstConn();

  // Update all net bounding boxes
  computeAllNetBBoxes();

  _iter = 0;
  _wl   = computeHPWL(_param.maxNetDegree);
  if (_param.verbose > 0) {
    auto wl = computeHPWL();
    openparfPrint(kInfo,
                  "Pre-ISM HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                  _param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y(),
                  _param.xWirelenWt,
                  wl.x(),
                  _param.yWirelenWt,
                  wl.y());
    openparfPrint(kInfo,
                  "ISM Iter %3u: Approx. HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                  _iter,
                  _param.xWirelenWt * _wl.x() + _param.yWirelenWt * _wl.y(),
                  _param.xWirelenWt,
                  _wl.x(),
                  _param.yWirelenWt,
                  _wl.y());
  }
}

/// Update all net bounding boxes
void ISMSolver::computeAllNetBBoxes() {
  for (auto &net : _netArray) {
    net.bbox.set(kRealTypeMax, kRealTypeMax, kRealTypeMin, kRealTypeMin);
    for (IndexType pinId : net.pinIdArray) {
      const auto &pin = _pinArray[pinId];
      auto        loc = getLoc(pin.instId);
      net.bbox.encompass(loc + pin.offset);
    }
  }
}

/// For each instance, initialize the set of instnaces that it is connected to
void ISMSolver::initInstConn() {
  ByteVector added(_numPhyInsts, 0);
  for (IndexType instId = 0; instId < _numPhyInsts; ++instId) {
    // Mark connected instances
    auto &inst    = _instArray[instId];
    added[instId] = 1;
    for (IndexType i : inst.pinIdArray) {
      const auto &net = _netArray[_pinArray[i].netId];
      if (net.numPins() > _param.maxNetDegree) {
        continue;
      }
      for (IndexType j : net.pinIdArray) {
        IndexType id = _pinArray[j].instId;
        if (!added[id]) {
          added[id] = 1;
          inst.conn.push_back(id);
        }
      }
    }
    // Reset added marks
    added[instId] = 0;
    for (IndexType i : inst.conn) {
      added[i] = 0;
    }
  }
}

/// Compute instance ISM priority, we give instances with less number moves higher priority
void ISMSolver::computeInstancePriority() {
  // Reset all buckets
  for (auto &bkt : _movBuckets) {
    bkt.clear();
  }

  // Find the maximum number of moves among all instances
  IndexType maxNumMovs = 0;
  for (IndexType i : _priority) {
    maxNumMovs = std::max(maxNumMovs, _instArray[i].numMovs);
  }

  // Assign instances to buckets based on their #moves
  _movBuckets.resize(maxNumMovs + 1);
  for (IndexType i : _priority) {
    _movBuckets[_instArray[i].numMovs].push_back(i);
  }

  // Concat all bucket to get the final sorting result
  auto it = _priority.begin();
  for (const auto &bkt : _movBuckets) {
    std::copy(bkt.begin(), bkt.end(), it);
    it += bkt.size();
  }
}

/// Build a set of independent "independent sets" that can be placed independently
/// Two independent sets are said to be independent if they do not share any common
/// nets that are larger than a given threshold
/// In this sense, by moving all nodes in this set of independent "independent sets",
/// in each net that smaller than the threshold, at most one node will be moved
void ISMSolver::buildIndependentIndepSets(IndepSet &indepSet, std::vector<IndexVector> &sets) {
  // Update instance ISM priority
  computeInstancePriority();

  // Reset independet marks for all instances
  std::fill(indepSet.dep.begin(), indepSet.dep.end(), 0);

  // Construct independent sets
  sets.clear();
  for (IndexType instId : _priority) {
    if (!indepSet.dep[instId]) {
      // Reset the independent set
      indepSet.set.clear();
      indepSet.type  = kIndexTypeMax;
      indepSet.cksr  = kIndexTypeMax;
      indepSet.ce[0] = kIndexTypeMax;
      indepSet.ce[1] = kIndexTypeMax;

      buildIndepSet(_instArray[instId], indepSet);
      sets.emplace_back(indepSet.set);
    }
  }
}

/// Build an independent set from a given seed instance
/// @param  seed      the seed instance
/// @param  indepSet  the empty independet set building helper
void ISMSolver::buildIndepSet(const ISMInstance &seed, IndepSet &indepSet) {
  // Add the seed instance into the independent set
  addInstToIndepSet(seed, 0, indepSet);
  indepSet.type = seed.type;

  // Spirally access the site map around the seed and grow the independent set
  // The access in Y must align with _param.siteYInterval
  openparfAssert(!seed.sol.isFixed());
  auto          initXYtmp = _siteMap.indexToXY(seed.sol.siteId);
  XY<IndexType> initXY(initXYtmp.x(), initXYtmp.y());
  initXY.setY(floorAlign(initXY.y(), _param.siteYInterval));
  auto beg = _spiralAccessor.begin(0);
  auto end = _spiralAccessor.end(_param.maxRadius);
  for (auto it = beg; it != end; ++it) {
    XY<IndexType> xy(initXY.x() + it->x() * _param.siteXStep, initXY.y() + it->y() * _param.siteYInterval);
    if (_bndBox.contain(xy)) {
      for (IndexType i = 0; i < _param.siteYInterval; ++i) {
        const auto &site = _siteMap(xy.x(), xy.y() + i);
        if (site.instId == kIndexTypeMax) {
          continue;
        }
        // Note that:
        //   fes 0: feasible w/o CE flipping
        //   fes 1: feasible w/ CE flipping
        //   fes 2: infeasible
        const auto &inst = _instArray[site.instId];
        IndexType   fes  = instToIndepSetFeasibility(inst, indepSet);
        if (fes == 2) {
          continue;
        }
        addInstToIndepSet(inst, fes, indepSet);
        if (indepSet.set.size() >= _param.maxIndepSetSize) {
          return;
        }
      }
    }
  }
}

/// Get the feasibility of adding the given instance into the independent set
/// @param   inst  the instance to be added
/// @return        0: feasible w/o CE flipping, 1: feasible w/ CE flipping, 2: infeasible
IndexType ISMSolver::instToIndepSetFeasibility(const ISMInstance &inst, const IndepSet &indepSet) {
  // Infeasible if the instance's type is different from the independent set
  // or the instance is dependent to instances already in the independent set
  if (inst.type != indepSet.type || indepSet.dep[inst.id]) {
    return 2;
  }

  openparfAssert(!inst.sol.isFixed());
  IndexType csId = _siteMap[inst.sol.siteId].ctrlSetId;
  if (csId == kIndexTypeMax) {
    // This instance does not subject to control set constraint
    return 0;
  }

  // The site control set and the independent set control set must be compatible
  const auto &cs = _ctrlSetArray[csId];
  // Check CK/SR compatibility
  if (cs.cksr != kIndexTypeMax && indepSet.cksr != kIndexTypeMax && cs.cksr != indepSet.cksr) {
    return 2;
  }
  // Check CE compatibility
  // Note that the 2 CEs can be flipped, so we need to check both (ce[0], ce[1]) and (ce[1], ce[0])
  // If both cases are feasible, return the one with smaller CE demand
  IndexType pos = computeCEMergeCost(cs.ce[0], cs.ce[1], indepSet.ce[0], indepSet.ce[1]);
  IndexType neg = computeCEMergeCost(cs.ce[0], cs.ce[1], indepSet.ce[1], indepSet.ce[0]);
  if (pos == kIndexTypeMax && neg == kIndexTypeMax) {
    return 2;
  }
  return (pos <= neg ? 0 : 1);
}

/// Compute the cost of merging two CE sets (a0, a1) and (b0, b1)
/// We define the CE merging cost as following:
//    1) if a0(a1) == b0(b1) cost += 0
//    2) if a0(a1) != b0(b1), but one of a0(a1) and b0(b1) is empty, cost += 1
//    3) otherwise, the two CE sets are not compatible, return cost = kIndexTypeMax
/// If the 2 CE sets are not compatible, return kIndexTypeMax
IndexType ISMSolver::computeCEMergeCost(IndexType a0, IndexType a1, IndexType b0, IndexType b1) const {
  IndexType cost = 0;
  if (a0 == b0) {
    cost += 0;
  } else if (a0 == kIndexTypeMax || b0 == kIndexTypeMax) {
    cost += 1;
  } else {
    return kIndexTypeMax;
  }
  if (a1 == b1) {
    cost += 0;
  } else if (a1 == kIndexTypeMax || b1 == kIndexTypeMax) {
    cost += 1;
  } else {
    return kIndexTypeMax;
  }
  return cost;
}

/// Add a given instance into the independent set
/// This function assumes adding the instance to the independent set is legal
/// @param  inst      the instance to be added
/// @param  ceFilp    whether the CEs in the instance need to be flipped to match the independent set control set
/// @param  indepSet  contains the results
void ISMSolver::addInstToIndepSet(const ISMInstance &inst, IndexType ceFlip, IndepSet &indepSet) {
  // Mark all instances that connect to the instance as dependent
  indepSet.dep[inst.id] = 1;
  for (IndexType i : inst.conn) {
    indepSet.dep[i] = 1;
  }

  openparfAssert(!inst.sol.isFixed());
  const auto &site = _siteMap[inst.sol.siteId];
  if (site.ctrlSetId != kIndexTypeMax) {
    // Here we also set instances in the same control set as dependent for the following two reasons:
    //   1) If we move two instances in the same site control set, even if they are independent,
    //      two other instances may compatible with this site control set at the same time, but we cannot guarantee
    //      that putting them into the same control set together is still legal. One simple example is that,
    //      there are 4 BLEs, A, B have the same control sets, and C and D have different control sets to each other and
    //      A and B. Assume site control set 0 only contains A and B, and site control set 1 and 2 only contain C and D,
    //      respectively. By construction, we will ripup all these 4 BLEs from their belonging half slices, so assigning
    //      any BLE to any slots in these site control set is legal, but putting C and D into site control set 0 at the
    //      same time is not legal
    //   2) By only moving one instance in each site control set, we can avoid mutex in multi-threaded mode,
    //      since any entry in each site control set will be modified by at most one thread at a time
    openparfAssert(!inst.sol.isFixed());
    XY<IndexType> siteXY(_siteMap.indexToXY(inst.sol.siteId));
    IndexType     yBeg = floorAlign(siteXY.y(), _param.numSitePerCtrlSet);
    IndexType     yEnd = yBeg + _param.numSitePerCtrlSet;
    for (IndexType y = yBeg; y < yEnd; ++y) {
      IndexType id = _siteMap(siteXY.x(), y).instId;
      if (id != kIndexTypeMax) {
        indepSet.dep[id] = 1;
      }
    }

    // Update the control set of the independent set
    // Note that we here need to add the site (NOT instance) control set of the instance into the independent set
    auto &cs       = _ctrlSetArray[site.ctrlSetId];
    indepSet.cksr  = (indepSet.cksr == kIndexTypeMax ? cs.cksr : indepSet.cksr);
    indepSet.ce[0] = (indepSet.ce[0] == kIndexTypeMax ? cs.ce[ceFlip] : indepSet.ce[0]);
    indepSet.ce[1] = (indepSet.ce[1] == kIndexTypeMax ? cs.ce[1 - ceFlip] : indepSet.ce[1]);

    // Update the control set of the site without the instance
    // Update the control set counts
    cs.cksrCount -= (inst.cksr == kIndexTypeMax ? 0 : 1);
    cs.ceCount[0] -= (inst.ce[0] == kIndexTypeMax ? 0 : 1);
    cs.ceCount[1] -= (inst.ce[1] == kIndexTypeMax ? 0 : 1);
    // Set the control set to empty if the count becomes 0
    cs.cksr  = (cs.cksrCount == 0 ? kIndexTypeMax : cs.cksr);
    cs.ce[0] = (cs.ceCount[0] == 0 ? kIndexTypeMax : cs.ce[0]);
    cs.ce[1] = (cs.ceCount[1] == 0 ? kIndexTypeMax : cs.ce[1]);
  }

  // Add the instance into the independent set
  indepSet.set.push_back(inst.id);
}

/// For each instance in the independent set, compute the bounding box of incident nets without considering the instance
void ISMSolver::computeNetBBoxes(const IndexVector &set, ISMMemory &mem) const {
  mem.bboxes.clear();
  mem.offset.clear();
  mem.netIds.clear();
  mem.ranges.clear();

  mem.ranges.push_back(0);
  for (IndexType idx = 0; idx < set.size(); ++idx) {
    IndexType   inst_id = set[idx];
    const auto &inst    = _instArray[set[idx]];
    auto        loc     = getLoc(inst_id);
    for (IndexType pinId : inst.pinIdArray) {
      const auto &pin = _pinArray[pinId];
      const auto &net = _netArray[pin.netId];
      if (net.numPins() > _param.maxNetDegree) {
        continue;
      }
      // Get the net bounding box without considering this instance
      mem.bboxes.emplace_back(net.bbox);
      mem.offset.emplace_back(pin.offset);
      mem.netIds.emplace_back(pin.netId);
      auto &bbox = mem.bboxes.back();

      // We require that there are no two pins in the same instance and also in the same net
      loc        = loc + pin.offset;
      if (loc.x() == bbox.xl()) {
        bbox.setXL(kRealTypeMax);
        for (IndexType pId : net.pinIdArray) {
          if (pId != pinId) {
            const auto &p = _pinArray[pId];
            auto        s = getLoc(p.instId);
            bbox.setXL(std::min(bbox.xl(), s.x() + p.offsetX()));
          }
        }
      }
      if (loc.y() == bbox.yl()) {
        bbox.setYL(kRealTypeMax);
        for (IndexType pId : net.pinIdArray) {
          if (pId != pinId) {
            const auto &p = _pinArray[pId];
            auto        s = getLoc(p.instId);
            bbox.setYL(std::min(bbox.yl(), s.y() + p.offsetY()));
          }
        }
      }
      if (loc.x() == bbox.xh()) {
        bbox.setXH(kRealTypeMin);
        for (IndexType pId : net.pinIdArray) {
          if (pId != pinId) {
            const auto &p = _pinArray[pId];
            auto        s = getLoc(p.instId);
            bbox.setXH(std::max(bbox.xh(), s.x() + p.offsetX()));
          }
        }
      }
      if (loc.y() == bbox.yh()) {
        bbox.setYH(kRealTypeMin);
        for (IndexType pId : net.pinIdArray) {
          if (pId != pinId) {
            const auto &p = _pinArray[pId];
            auto        s = getLoc(p.instId);
            bbox.setYH(std::max(bbox.yh(), s.y() + p.offsetY()));
          }
        }
      }
    }
    mem.ranges.push_back(mem.bboxes.size());
  }
}

/// Fill cost matrix for ISM
void ISMSolver::computeCostMatrix(const IndexVector &set, ISMMemory &mem) const {
  mem.costMtx.resize(set.size(), set.size());
  for (IndexType i = 0; i < set.size(); ++i) {
    const auto &mates = _instArray[set[i]].mates;
    for (IndexType j = 0; j < set.size(); ++j) {
      // Compute the moving cost (HPWL incr.) of moving the i-th instance to the position of the j-th instance
      RealType cost = 0;
      openparfAssert(!_instArray[set[j]].sol.isFixed());
      const auto &site    = _siteMap[_instArray[set[j]].sol.siteId];
      // If not clock region or half column compatiable, then the cost would be set to INTMAX
      bool        illegal = false;
      if (_honorClockConstraints) {
        for (IndexType clk : _instArray[set[i]].ckArray) {
          uint32_t sx = uint32_t(site.x());
          uint32_t sy = uint32_t(site.y());
          if (!_isCKAllowedInSite(clk, sx, sy)) {
            mem.costMtx(i, j) = std::numeric_limits<FlowIntType>::max();
            illegal           = true;
            break;
          }
        }
      }
      if (illegal) continue;
      for (IndexType k = mem.ranges[i]; k < mem.ranges[i + 1]; ++k) {
        XY<RealType> loc(site.loc + mem.offset[k]);
        const auto & bbox = mem.bboxes[k];
        RealType     wt   = _netArray[mem.netIds[k]].weight;
        cost += wt * (_param.xWirelenWt * std::max(std::max(bbox.xl() - loc.x(), loc.x() - bbox.xh()), (RealType) 0.0) +
                      _param.yWirelenWt * std::max(std::max(bbox.yl() - loc.y(), loc.y() - bbox.yh()), (RealType) 0.0));
      }
      // Compute the mate credit (credit given to assignment achieving internel nets)
      RealType mateCredit = 0;
      if (site.mateId != kIndexTypeMax) {
        IndexType id = _siteMap[site.mateId].instId;
        for (IndexType m : mates) {
          mateCredit += (m == id ? _param.mateCredit : 0);
        }
      }
      mem.costMtx(i, j) = (cost - mateCredit) * _param.flowCostScale;
    }
  }
}

/// Perform ISM kernel
void ISMSolver::computeMatching(ISMMemory &mem) const {
  using GraphType  = lemon::ListDigraph;
  using SolverType = lemon::NetworkSimplex<GraphType, FlowIntType>;
  using ResultType = typename SolverType::ProblemType;

  auto &graph      = *(mem.graphPtr);
  auto &lNodes     = mem.lNodes;
  auto &rNodes     = mem.rNodes;
  auto &lArcs      = mem.lArcs;
  auto &rArcs      = mem.rArcs;
  auto &mArcs      = mem.mArcs;
  auto &mArcPairs  = mem.mArcPairs;

  graph.clear();
  lNodes.clear();
  rNodes.clear();
  lArcs.clear();
  rArcs.clear();
  mArcs.clear();
  mArcPairs.clear();

  // Flow cost/capacity maps
  lemon::ListDigraph::ArcMap<FlowIntType> capLo(graph);
  lemon::ListDigraph::ArcMap<FlowIntType> capHi(graph);
  lemon::ListDigraph::ArcMap<FlowIntType> costMap(graph);

  // Source and target nodes
  auto                                    s = graph.addNode();
  auto                                    t = graph.addNode();

  // Add arcs between source(right) and left(target)
  IndexType                               n = mem.costMtx.xSize();
  for (IndexType i = 0; i < n; ++i) {
    // Source to left
    lNodes.emplace_back(graph.addNode());
    lArcs.emplace_back(graph.addArc(s, lNodes.back()));
    capLo[lArcs.back()]   = 0;
    capHi[lArcs.back()]   = 1;
    costMap[lArcs.back()] = 0;

    // Right to target
    rNodes.emplace_back(graph.addNode());
    rArcs.emplace_back(graph.addArc(rNodes.back(), t));
    capLo[rArcs.back()]   = 0;
    capHi[rArcs.back()]   = 1;
    costMap[rArcs.back()] = 0;
  }

  // Generate arcs between left and right
  for (IndexType l = 0; l < n; ++l) {
    for (IndexType r = 0; r < n; ++r) {
      if (mem.costMtx(l, r) == std::numeric_limits<FlowIntType>::max()) {
        continue;
      }
      mArcs.emplace_back(graph.addArc(lNodes[l], rNodes[r]));
      mArcPairs.emplace_back(l, r);
      costMap[mArcs.back()] = mem.costMtx(l, r);
      capLo[mArcs.back()]   = 0;
      capHi[mArcs.back()]   = 1;
    }
  }

  // Perform network simplex algorithm to solve the min-cost bipartite matching
  SolverType ns(graph);
  ns.stSupply(s, t, n);
  ns.lowerMap(capLo).upperMap(capHi).costMap(costMap);
  ResultType res = ns.run();
  openparfAssertMsg(res == ResultType::OPTIMAL, "ISM DP MCF not feasible\n");

  // Collect the ISM solution
  mem.sol.resize(n);
  for (IndexType i = 0; i < mArcs.size(); ++i) {
    if (ns.flow(mArcs[i])) {
      const auto &p    = mArcPairs[i];
      mem.sol[p.first] = p.second;
    }
  }
}

/// Realize the ISM solution
void ISMSolver::realizeMatching(const IndexVector &set, ISMMemory &mem) {
  // Cache the site ID of each instance before moving
  auto &siteIds = mem.idxVec;
  siteIds.resize(set.size());
  for (IndexType i = 0; i < set.size(); ++i) {
    siteIds[i] = _instArray[set[i]].sol.siteId;
  }

  // Move instances to the locations of their matching instances
  for (IndexType i = 0; i < set.size(); ++i) {
    // Update instance locations
    auto &inst      = _instArray[set[i]];
    inst.sol.siteId = siteIds[mem.sol[i]];
    ++inst.numMovs;

    // Update site
    auto &site  = _siteMap[inst.sol.siteId];
    site.instId = inst.id;

    // Update site control set
    if (site.ctrlSetId != kIndexTypeMax) {
      auto &cs = _ctrlSetArray[site.ctrlSetId];
      if (inst.cksr != kIndexTypeMax) {
        cs.cksr = inst.cksr;
        ++cs.cksrCount;
      }
      // Get the cheapest CE fitting
      IndexType posCost = computeCEMergeCost(cs.ce[0], cs.ce[1], inst.ce[0], inst.ce[1]);
      IndexType negCost = computeCEMergeCost(cs.ce[0], cs.ce[1], inst.ce[1], inst.ce[0]);
      if (posCost > negCost) {
        std::swap(inst.ce[0], inst.ce[1]);
        inst.sol.ceFlip = !inst.sol.ceFlip;
      }
      for (IndexType k : {0, 1}) {
        if (inst.ce[k] != kIndexTypeMax) {
          cs.ce[k] = inst.ce[k];
          ++cs.ceCount[k];
        }
      }
    }

    // Update net bounding boxes
    for (IndexType k = mem.ranges[i]; k < mem.ranges[i + 1]; ++k) {
      auto &bbox = _netArray[mem.netIds[k]].bbox;
      bbox       = mem.bboxes[k];
      bbox.encompass(site.loc + mem.offset[k]);
    }
  }
}

/// Return true if the stopping condition holds
bool ISMSolver::stopCondition() {
  if (_iter && (_iter % _param.batchSize == 0)) {
    auto wl = computeHPWL(_param.maxNetDegree);
    if (_param.verbose > 0) {
      openparfPrint(kInfo,
                    "ISM Iter %3u: Approx. HPWL = %.6E (%.1lf * %.6E + %.1lf * %.6E)\n",
                    _iter,
                    _param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y(),
                    _param.xWirelenWt,
                    wl.x(),
                    _param.yWirelenWt,
                    wl.y());
    }
    RealType improv = 1.0 - (_param.xWirelenWt * wl.x() + _param.yWirelenWt * wl.y()) /
                                    (_param.xWirelenWt * _wl.x() + _param.yWirelenWt * _wl.y());
    if (improv < _param.minBatchImprov) {
      return true;
    }
    _wl = wl;
  }
  ++_iter;
  return false;
}

/// Compute HPWL that ignore nets having degree larger than a given threshold
XY<RealType> ISMSolver::computeHPWL(IndexType maxNetDegree) const {
  XY<RealType> res(0, 0);
  for (const auto &net : _netArray) {
    if (net.numPins() <= maxNetDegree) {
      res.set(res.x() + net.weight * net.bbox.width(), res.y() + net.weight * net.bbox.height());
    }
  }
  return res;
}

}   // namespace ism_dp
OPENPARF_END_NAMESPACE
