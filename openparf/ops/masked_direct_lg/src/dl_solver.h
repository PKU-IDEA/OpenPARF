/**
 * File              : dl_solver.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_DIRECT_LG_SRC_DL_SOLVER_H_
#define OPENPARF_OPS_DIRECT_LG_SRC_DL_SOLVER_H_

// C system headers
#include <omp.h>

// C++ standard library headers
#include <memory>
#include <utility>
#include <vector>

// other libraries headers
#include "boost/container/flat_map.hpp"
#include "lemon/list_graph.h"
#include "lemon/network_simplex.h"
#include "lemon/smart_graph.h"

// project headers
#include "container/fixed_size_pq.hpp"
#include "container/spiral_accessor.hpp"
#include "database/clock_availability.h"
#include "util/util.h"

// local headers
#include "ops/masked_direct_lg/src/dl_problem.h"

OPENPARF_BEGIN_NAMESPACE

namespace masked_direct_lg {

using container::FixedSizePQ;
using container::SpiralAccessor;

/// The kernel to solve direct legalization (DL) problem
class DLSolver {
 private:
  using FlowIntType   = std::int64_t;
  using IndexVector   = std::vector<IndexType>;
  using RealVector    = std::vector<RealType>;
  using IndexMap      = boost::container::flat_map<IndexType, IndexType>;
  using IndexMap2     = boost::container::flat_map<IndexType, IndexMap>;
  using IndexRealPair = std::pair<IndexType, RealType>;

 public:
  /// Class to represent the final DL solution of an instance
  struct DLInstanceSol {
    DLInstanceSol() = default;
    void set(IndexType a, IndexType b) {
      siteId = a;
      z      = b;
    }

    IndexType siteId = kIndexTypeMax;   // ID of the assigned site
    IndexType z      = kIndexTypeMax;   // Z value in the site
  };

 private:
  // Enum type for DL status after synchronization
  enum class DLStatus : Byte {
    LEGAL,     // All instances are legal
    ILLEGAL,   // Not all instances are legal and there is no active candidate
    ACTIVE     // Not all instances are legal and there are still active candidates
  };

  enum class LutCandidateType : Byte { UNDECIDED, LUT, LRAM, SHIFT };
  /// Class for instances (e.g., LUT/FF) used in DL
  struct DLInstance {
    /// Data members that need to be synchronously updated in each DL iteration
    struct SyncData {
      IndexType detSiteId       = kIndexTypeMax;   // The determined site ID
      IndexType bestSiteId      = kIndexTypeMax;   // The best site ID in a DL iteration
      RealType  bestScoreImprov = kRealTypeMin;    // The score improv of the bset site in a DL iteration
    };

    DLInstance() : type(DLInstanceType::DONTCARE) {}
    ~DLInstance() {}

    void initLock() { omp_init_lock(&lock); }
    void destroyLock() { omp_destroy_lock(&lock); }

    void updateBestSite(IndexType siteId, RealType scoreImprov) {
      if (curr.detSiteId != kIndexTypeMax) {
        return;
      }
      omp_set_lock(&lock);
      if (scoreImprov == next.bestScoreImprov) {
        if (siteId < next.bestSiteId) {
          next.bestSiteId = siteId;
        }
      } else if (scoreImprov > next.bestScoreImprov) {
        next.bestSiteId      = siteId;
        next.bestScoreImprov = scoreImprov;
      }
      omp_unset_lock(&lock);
    }

    IndexType      numPins() const { return pinIdArray.size(); }
    RealType       x() const { return loc.x(); }
    RealType       y() const { return loc.y(); }

    IndexType      origId = kIndexTypeMax;              // The original index of this instance when it is constructed
    IndexType      id     = kIndexTypeMax;              // Instance ID
    DLInstanceType type   = DLInstanceType::DONTCARE;   // Instance type
    bool           isFixed;
    XY<RealType>   loc;                      // Instance location
    IndexType      dem    = 0;               // Resource demand, #inputs for LUT
    IndexType      cksr   = kIndexTypeMax;   // ID of CK/SR for FF
    IndexType      ce     = kIndexTypeMax;   // ID of CE for FF
    RealType       weight = 1.0;             // Weight for centroid computation
    RealType       area   = 0.0;   // Instance relative area indiciating the level of difficulty to legalize it
    IndexVector    pinIdArray;     // IDs of incident pins of this instance
    IndexVector    precluster;     // The set of DL-instances that need to be moved together, this list is sorted
    SyncData       curr;           // Sync data of the current iteration, this cannot be changed during a DL iteration
    SyncData       next;           // Sync data of the next iteration, this can be changed during a DL iteration
    omp_lock_t     lock;           // Mutex lock for multi-threading
    DLInstanceSol  sol;            // The location (site ID, z) after DL
  };

  /// Class for nets used in DL
  struct DLNet {
    IndexType     numPins() const { return pinIdArray.size(); }

    IndexType     origId = kIndexTypeMax;   // The original index of this net when it is constructed
    IndexType     id     = kIndexTypeMax;   // Net ID
    IndexVector   pinIdArray;               // IDs of pins in this net
    IndexVector   pinIdArrayX;              // IDs of pins in this net sorted by their X from low to high
    IndexVector   pinIdArrayY;              // IDs of pins in this net sorted by their Y from low to high
    RealType      weight = 0;               // Wirelength weight
    Box<RealType> bbox;                     // Net bounding box
  };

  /// Class for pins used in DL
  struct DLPin {
    RealType     offsetX() const { return offset.x(); }
    RealType     offsetY() const { return offset.y(); }

    IndexType    origId = kIndexTypeMax;   // The original index of this pin when it is constructed
    IndexType    instId = kIndexTypeMax;   // ID of the incident instance
    IndexType    netId  = kIndexTypeMax;   // ID of the incident net
    DLPinType    type;                     // Pin type
    XY<RealType> offset;                   // Pin offset w.r.t. instance
  };

  /// Class to store a candidate
  struct Candidate {
    /// Class to store the detailed information of a candidate
    /// It includs instance IDs, their slot assignment, and control set
    struct Implementation {
      explicit Implementation(IndexType num_LUTs, IndexType num_FFs) {
        lut.resize(num_LUTs);
        ff.resize(num_FFs);
        reset();
      }
      void reset() {
        candidate_type = LutCandidateType::UNDECIDED;
        std::fill(lut.begin(), lut.end(), kIndexTypeMax);
        std::fill(ff.begin(), ff.end(), kIndexTypeMax);
        std::fill(cksr.begin(), cksr.end(), kIndexTypeMax);
        std::fill(ce.begin(), ce.end(), kIndexTypeMax);
      }
      std::vector<IndexType>   lut;   // LUT slot assignment
      LutCandidateType         candidate_type;
      std::vector<IndexType>   ff;     // FF slot assignment
      std::array<IndexType, 2> cksr;   // Clock IDs
      std::array<IndexType, 4> ce;     // Clock enable IDs
    };

    /// Comparator for Candidate objects
    struct Comparator {
      // Return true if a has higher priority (score) than b
      // If the two candidiates have the same score, return the one has larger volumn
      // If the two candidates also have the same volumn, break the tie using instance IDs
      bool operator()(const Candidate &a, const Candidate &b) const {
        if (a.score == b.score) {
          return (a.sig.size() != b.sig.size() ? a.sig.size() > b.sig.size() : a.sig < b.sig);
        }
        return a.score > b.score;
      }
    };
    bool operator==(const Candidate &rhs) const { return sig == rhs.sig; }
    bool operator!=(const Candidate &rhs) const { return !(*this == rhs); }

    Candidate(IndexType num_LUTs, IndexType num_FFs)
        : sig(),
          impl(num_LUTs, num_FFs),
          score(0.0),
          siteId(kIndexTypeMax) {}

    IndexVector    sig;                      // Sorted instance IDs in this candidate
    Implementation impl;                     // Implementation details of this candidate
    RealType       score  = 0.0;             // Score of this candidate
    IndexType      siteId = kIndexTypeMax;   // The ID of site that this candidate being assigned to
  };

  using CandidatePQ = FixedSizePQ<Candidate, Candidate::Comparator>;

  /// Class for sites (e.g., SLICE) used in DL
  struct DLSite {
    /// Data member that need to be synchronously updated in each DL iteration
    struct SyncData {
      explicit SyncData(IndexType pqSize) : pq(pqSize) {}

      CandidatePQ            pq;           // PQ for candidates
      std::vector<Candidate> scl;          // Seed candidate list
      IndexType              stable = 0;   // Number of iterations since last pq.top change
    };

    explicit DLSite(IndexType pqSize, IndexType num_LUTs, IndexType num_FFs)
        : curr(pqSize),
          next(pqSize),
          det(num_LUTs, num_FFs) {}
    RealType     x() const { return loc.x(); }
    RealType     y() const { return loc.y(); }
    IndexType    numNbrGroups() const { return nbrRanges.empty() ? 0 : nbrRanges.size() - 1; }

    IndexType    id   = kIndexTypeMax;          // Site ID
    DLSiteType   type = DLSiteType::DONTCARE;   // Site type
    XY<RealType> loc;                           // Site location
    IndexVector  nbr;                           // Sorted IDs of active neighbor instances under consideration
    IndexVector  nbrList;   // IDs of all neighbor instances (active and inactive), sorted by (nbr, site) distance from
                            // low to high
    IndexVector  nbrRanges;         // [arr[i], arr[i+1]) is the range of i-th neighbor group
    IndexType    nbrGroupIdx = 0;   // The largest neighbor group index that has been added to 'nbr'
    SyncData     curr;              // Sync data of the current iteration, this cannot be changed during a DL iteration
    SyncData     next;              // Sync data of the next iteration, this can be changed during a DL iteration
    Candidate    det;               // Determined candidate
  };

  /// Candidate for ripup-and-legalize
  struct RipupCandidate {
    RipupCandidate(IndexType num_LUTs, IndexType num_FFs)
        : siteId(kIndexTypeMax),
          score(kRealTypeMin),
          legal(false),
          cand(num_LUTs, num_FFs) {}
    // If a < b, then a has higher priority than b
    bool      operator<(const RipupCandidate &rhs) const { return (legal == rhs.legal ? score > rhs.score : legal); }

    IndexType siteId = kIndexTypeMax;   // The site to be ripped up
    RealType  score  = kRealTypeMin;    // The ripup score to determine the priority of this candidate
    bool      legal  = false;           // If this is true, then we don't need to rip up the site
    Candidate cand;                     // If legal == true, this contains the legal candidate
  };

  /// Class for memory used by slot assignment
  struct SlotAssignMemory {
    SlotAssignMemory() : graphPtr(new lemon::ListGraph()), biGraphPtr(new lemon::SmartDigraph()) {}

    // Class to store a BLE candidate
    struct BLE {
      std::array<IndexType, 2> lut    = {{kIndexTypeMax, kIndexTypeMax}};
      std::array<IndexType, 2> ff     = {{kIndexTypeMax, kIndexTypeMax}};
      RealType                 score  = 0;
      RealType                 improv = 0;
    };

    Candidate                             *candPtr;   // The candidate to work on
    IndexVector                            lut;       // IDs of all LUTs in 'impl'
    IndexVector                            ff;        // IDs of all FFs in 'impl'
    std::array<IndexType, 2>               cksr;      // Pre-assigned CK/SR
    std::array<IndexType, 4>               ce;        // Pre-assigned CE
    std::vector<BLE>                       bleS;      // single-LUT BLE candidates before LUT pairing
    std::vector<BLE>                       bleP;      // paired-LUT BLE candidates before LUT pairing
    std::vector<BLE>                       ble;       // BLE candidate after LUT pairing
    RealVector                             scores;    // Score of each BLE slot used for FF assigment

    // Max-weighted matching for LUT pairing
    // Since lemon::ListGraph is not constructible, we need to wrap it with std::unique_ptr
    std::unique_ptr<lemon::ListGraph>      graphPtr;
    std::vector<lemon::ListGraph::Node>    nodes;
    std::vector<lemon::ListGraph::Edge>    edges;


    // Matching for LUTs & FFs assignment
    std::unique_ptr<lemon::SmartDigraph>   biGraphPtr;
    std::vector<lemon::SmartDigraph::Node> leftNodes;
    std::vector<lemon::SmartDigraph::Node> rightNodes;

    // Common memory buffers
    IndexVector                            idxVec;
    IndexMap                               idxMap;
  };

  /// Class for memory used by each thread
  struct ThreadMemory {
    ThreadMemory() : graphPtr(new lemon::ListGraph()) {}

    // Shared indenx vectors
    IndexVector                                  idxVecA;
    IndexVector                                  idxVecB;

    // For signature generation
    IndexVector                                  sigBuf;

    // For max-matching based LUT assignment
    // Since lemon::ListGraph is not constructible, we need to wrap it with std::unique_ptr
    std::unique_ptr<lemon::ListGraph>            graphPtr;
    std::vector<lemon::ListGraph::Node>          nodes;
    std::vector<lemon::ListGraph::Edge>          edges;
    std::vector<std::pair<IndexType, IndexType>> edgePairs;

    // For slot assignment
    SlotAssignMemory                             slotAssignMem;
  };

  // Inspired by the implementation in UTPlaceFX
  struct HalfColumnRegionScoreBoard {
    explicit HalfColumnRegionScoreBoard(IndexType numClockNets)
        : isAvail(numClockNets, 1),
          clockCost(numClockNets, 0.0),
          isFixed(numClockNets, 0) {}
    // All the clock are clock net indexes, not clock net id.
    std::vector<Byte>     isAvail;     // Record if a clock (index) is allowed in this HC
    std::vector<RealType> clockCost;   // The cost of forbidding a clock in this HC
    std::vector<Byte> isFixed;   // isFixed[i] records if a fix inst (i.e. a previously legalized DSP/RAM) is in this HC
                                 // and it contains clk i
  };


 public:
  DLSolver(const DLProblem                         &prob,
           const DirectLegalizeParam               &param,
           ClockAvailCheckerType<RealType>          legality_check_functor,
           LayoutXy2GridIndexFunctorType<RealType>  xy_to_half_column_functor,
           const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
           IndexType                                num_threads);
  void run();

  bool clkAvailableAtHalfColumn(int32_t clk, int32_t half_column_id) const {
    return (_hcsbArray.at(half_column_id).isFixed.at(clk) == 1 ||
            _hcsbArray.at(half_column_id).clockCost.at(clk) > 1e-6);
  }


  // Get the final DL solution of a instance
  const DLInstanceSol &instSol(IndexType i) const { return _instArray[_instIdMapping[i]].sol; }

 private:
  IndexType    numInsts() const { return _instArray.size(); }
  IndexType    numNets() const { return _netArray.size(); }
  IndexType    numPins() const { return _pinArray.size(); }
  IndexType    numSites() const { return _siteMap.size(); }

  // Problem initialization
  void         buildDLProblem(const DLProblem &prob);
  void         buildNetlist(const DLProblem &prob);
  void         sortNetlist();
  void         sortNets(std::function<bool(IndexType, IndexType)> cmp);
  void         sortPins(std::function<bool(IndexType, IndexType)> cmp);
  void         sortInstances(std::function<bool(IndexType, IndexType)> cmp);
  void         buildSiteMap(const DLProblem &prob);
  void         initControlSets();
  void         initClockConstraints();

  // DL initialization
  void         initNets();
  void         preclustering();
  void         initSiteNeighbors();
  void         allocateMemory();

  // DL kernel functions
  void         runDLInit();
  void         runDLIteration(DLSite &site);
  DLStatus     runDLSync();
  bool         commitTopCandidate(DLSite &site);
  void         removeIncompatibleNeighbors(DLSite &site);
  void         removeInvalidCandidates(DLSite &site);
  void         removeCommittedNeighbors(DLSite &site);
  void         addNeighbors(DLSite &Site);
  void         createNewCandidates(DLSite &site);
  void         broadcastTopCandidate(const DLSite &site);

  // DL helping functions
  bool         addInstToCandidateIsFeasible(const DLInstance &inst, const Candidate &cand);
  bool         addInstToCandidateImpl(const DLInstance &inst, Candidate &cand);
  bool         addInstToCandidateImpl(const DLInstance                &instance,
                                      const Candidate::Implementation &impl,
                                      Candidate::Implementation       &res);
  bool         addFFToCandidateImpl(const DLInstance &ff, Candidate::Implementation &impl) const;
  bool         addLUTToCandidateImpl(const DLInstance &lut, Candidate::Implementation &impl) const;
  bool         fitLUTsToCandidateImpl(const IndexVector &luts, Candidate::Implementation &impl);
  RealType     computeCandidateScore(const Candidate &cand);
  RealType     computeWirelenImprov(const DLNet &net, const IndexVector &pins, const XY<RealType> &loc) const;
  XY<RealType> computeInstCentroid(const IndexVector &instIdArray) const;
  bool         instAndSiteAreCompatible(const DLInstance &instType, const DLSite &siteType) const;
  bool         candidateIsValid(const Candidate &cand) const;
  bool         twoLUTsAreCompatible(const DLInstance &lutA, const DLInstance &lutB) const;
  IndexType    findInstInputPinIndex(const DLInstance &inst, IndexType beg) const;
  bool         addInstToSignature(const DLInstance &inst, IndexVector &sig);

  // Post legalization
  bool         ripupLegalization();
  bool         ripupLegalizeInst(DLInstance &inst, RealType maxD);
  bool         createRipupCandidate(const DLInstance &inst, const DLSite &site, RipupCandidate &res);
  bool         ripupSiteAndLegalizeInstance(DLInstance &inst, DLSite &site, RealType maxD);
  bool         greedyLegalization();
  bool         greedyLegalizeInst(DLInstance &inst);
  void         cacheSolution();

  // Slot assignment
  void         runSlotAssign();
  void         initSlotAssign(Candidate &cand, SlotAssignMemory &mem);
  void         computeLUTScoreAndScoreImprov(SlotAssignMemory &mem);
  void         pairLUTs(SlotAssignMemory &mem);
  bool         checkFFsToBLEsFeasibility(SlotAssignMemory &mem, std::array<IndexType, 4> &availCE);
  void         assignLRAMsAndFFs(SlotAssignMemory &mem);
  void         assignLUTsAndFFs(SlotAssignMemory &mem);
  void         findBestFFs(const IndexVector              &ff,
                           const std::array<IndexType, 2> &cksr,
                           const std::array<IndexType, 4> &ce,
                           SlotAssignMemory::BLE          &ble);
  void     findBestFFs(const IndexVector &ff, IndexType cksr, IndexType ce0, IndexType ce1, SlotAssignMemory::BLE &ble);
  void     orderBLEs(SlotAssignMemory &mem);
  RealType computeBLEScore(IndexType lutA, IndexType lutB, IndexType ffA, IndexType ffB);
  IndexType    computeNumShareInputs(const DLInstance &a, const DLInstance &b) const;
  RealType     computeNetShareScore(const IndexVector &pins) const;
  IndexType    computeNumInternalNets(IndexType lutA, IndexType lutB, IndexType ffA, IndexType ffB) const;

  // Clock Region & Half column related
  void         initHalfColumnScoreBoardArray();
  bool         checkHalfColumnSatAndPruneClock();
  void         calculateHalfColumnScoreBoardCost();
  bool         instHalfColumnLegalAtSite(const DLInstance &inst, const DLSite &site) const;
  bool         instAndSiteClockCompatible(const DLInstance &inst, const DLSite &site) const;

  // Message printing
  XY<RealType> computeHPWL() const;
  void         printDLStatistics() const;
  void         printPostLegalizationStatistics() const;
  void         printDLStatus() const;

 private:
  const DirectLegalizeParam      &_param;
  SpiralAccessor                  _spiralAccessor;   // The spiral accessor to help add instance/site spatial neighbors
  Box<IntType>                    _bndBox;           // The bounding box of the placement region, helps spiral accessing

  std::vector<DLInstance>         _instArray;   // All instances
  std::vector<DLNet>              _netArray;    // All nets
  std::vector<DLPin>              _pinArray;    // All pins
  Vector2D<DLSite>                _siteMap;     // The site map

  IndexVector                     _instIdMapping;   // The instance original ID to sorted ID mapping
  IndexMap2                       _cksrMapping;     // Net ID to control set CK/SR ID mapping
  IndexMap                        _ceMapping;       // Net ID to control set CK/SR ID mapping

  // For DL iterations
  IndexType                       _dlIter = 0;       // Number of DL iterations finished
  std::vector<ThreadMemory>       _threadMemArray;   // Memory used by each thread

  IndexType                       _num_LUTs;   // #LUTs per CLB
  IndexType                       _num_FFs;    // #FFs per CLB

  // Clock Constraint Attributes
  IndexType                       _numLUTInst;   // number of LUT instances
  IndexType                       _numFFInst;    // number of FF instances
  IndexType                       _numSiteSLICE;
  ClockAvailCheckerType<RealType> _legality_check_functor;
  LayoutXy2GridIndexFunctorType<RealType>  _xy_to_half_column_functor;
  std::vector<HalfColumnRegionScoreBoard>  _hcsbArray;        // half column score board array
  const std::vector<std::vector<int32_t>> &_instClockIndex;   // Clock Indexes of instances
  IndexType                                _num_threads;      // number of threads
};

/// Compute the centroid of a set of instances
inline XY<RealType> DLSolver::computeInstCentroid(const IndexVector &instIdArray) const {
  XY<RealType> cen(0, 0);
  RealType     totalWt = 0;
  for (IndexType i : instIdArray) {
    const auto &inst = _instArray[i];
    cen.set(cen.x() + inst.loc.x() * inst.weight, cen.y() + inst.loc.y() * inst.weight);
    totalWt += inst.weight;
  }
  cen.set(cen.x() / totalWt, cen.y() / totalWt);
  return cen;
}

/// Check if a instance type and a site type are compatible
inline bool DLSolver::instAndSiteAreCompatible(const DLInstance &inst, const DLSite &site) const {
  if (inst.type == DLInstanceType::DONTCARE) {
    return false;
  }
  if ((inst.type == DLInstanceType::LRAM || inst.type == DLInstanceType::SHIFT) && site.type != DLSiteType::SLICEM)
    return false;
  if (site.type != DLSiteType::SLICEM && site.type != DLSiteType::SLICE) return false;
  return true;
}

inline bool DLSolver::instHalfColumnLegalAtSite(const DLInstance &inst, const DLSite &site) const {
  auto hc_id = _xy_to_half_column_functor(site.x(), site.y());
  for (auto clk : _instClockIndex[inst.origId]) {
    auto const &sb = _hcsbArray.at(hc_id);
    if (!sb.isAvail.at(clk)) return false;
  }
  return true;
}

inline bool DLSolver::instAndSiteClockCompatible(const DLInstance &inst, const DLSite &site) const {
  int32_t  inst_id = inst.origId;
  RealType site_x  = site.x();
  RealType site_y  = site.y();
  if (_param.honorClockRegionConstraint && !_legality_check_functor(inst_id, site_x, site_y)) {
    return false;
  }

  if (_param.honorHalfColumnConstraint && !instHalfColumnLegalAtSite(inst, site)) {
    return false;
  }
  return true;
}

/// Check if a candidate is valid in a site
inline bool DLSolver::candidateIsValid(const Candidate &cand) const {
  for (IndexType instId : cand.sig) {
    /// A candidate is invalid if any of its constituting instance has been assigned to another site
    IndexType detSiteId = _instArray[instId].curr.detSiteId;
    if (detSiteId != kIndexTypeMax && detSiteId != cand.siteId) {
      return false;
    }
  }
  return true;
}

/// Check if 2 LUTs can be fitted into the same BLE
inline bool DLSolver::twoLUTsAreCompatible(const DLInstance &lutA, const DLInstance &lutB) const {
  // If one of the two LUTs is 6-input LUT, they are not compatible
  if (!(lutA.type == DLInstanceType::LUT && lutB.type == DLInstanceType::LUT)) return false;
  if (lutA.dem == 6 || lutB.dem == 6) {
    return false;
  }

  // The total number of distinct input nets should be less than or equal to 5
  // Note that pin IDs in inst.pinIdArray are sorted in their incident net order
  IndexType numInputs = lutA.dem + lutB.dem;
  IndexType idxA      = findInstInputPinIndex(lutA, 0);
  IndexType idxB      = findInstInputPinIndex(lutB, 0);
  IndexType netA      = _pinArray[lutA.pinIdArray[idxA]].netId;
  IndexType netB      = _pinArray[lutB.pinIdArray[idxB]].netId;
  while (numInputs > 5) {
    if (netA < netB) {
      if ((idxA = findInstInputPinIndex(lutA, idxA + 1)) >= lutA.numPins()) {
        break;
      }
      netA = _pinArray[lutA.pinIdArray[idxA]].netId;
    } else if (netA > netB) {
      idxB = findInstInputPinIndex(lutB, idxB + 1);
      if (idxB >= lutB.numPins()) {
        break;
      }
      netB = _pinArray[lutB.pinIdArray[idxB]].netId;
    } else {
      // netA == netB
      --numInputs;
      if ((idxA = findInstInputPinIndex(lutA, idxA + 1)) >= lutA.numPins() ||
          (idxB = findInstInputPinIndex(lutB, idxB + 1)) >= lutB.numPins()) {
        break;
      }
      netA = _pinArray[lutA.pinIdArray[idxA]].netId;
      netB = _pinArray[lutB.pinIdArray[idxB]].netId;
    }
  }
  return numInputs <= 5;
}

/// Find the index of the first input pin in a given instance from the given index
/// Helping function for "twoLUTsAreCompatible"
inline IndexType DLSolver::findInstInputPinIndex(const DLInstance &inst, IndexType beg) const {
  while (beg < inst.numPins() && _pinArray[inst.pinIdArray[beg]].type != DLPinType::INPUT) {
    ++beg;
  }
  return beg;
}

/// Insert a instance into a signature and keep it sorted, return true on success
/// If the signature already contains the instance, return false
inline bool DLSolver::addInstToSignature(const DLInstance &inst, IndexVector &sig) {
  auto &m = _threadMemArray[omp_get_thread_num()].sigBuf;
  m.clear();

  auto itA  = sig.begin();
  auto itB  = inst.precluster.begin();
  auto endA = sig.end();
  auto endB = inst.precluster.end();
  while (itA != endA && itB != endB) {
    if (*itA < *itB) {
      m.push_back(*itA);
      ++itA;
    } else if (*itA > *itB) {
      m.push_back(*itB);
      ++itB;
    } else {
      // *itA == *itB
      return false;
    }
  }
  if (itA == endA) {
    m.insert(m.end(), itB, endB);
  } else {
    // itB == endB
    m.insert(m.end(), itA, endA);
  }
  sig.swap(m);
  return true;
}

/// Compute the score of a BLE (2 LUTs and 2FFs) for slot assignment
/// This function assumes the BLE is feasible
/// For now, this function is only called for slot assignment score computation
inline RealType DLSolver::computeBLEScore(IndexType lutA, IndexType lutB, IndexType ffA, IndexType ffB) {
  // The score consist of 3 parts
  //   (1) LUT input pin sharing score
  //   (2) internal connection (LUT.output -> FF.input nets that can be fully absorbed in a BLE) score
  //   (3) total number of FFs for tie-breaking

  // Compute (1) LUT input pin sharing score
  IndexType numLUTShareInputs = 0;
  if (lutA != kIndexTypeMax && lutB != kIndexTypeMax) {
    numLUTShareInputs = computeNumShareInputs(_instArray[lutA], _instArray[lutB]);
  }

  // Compute (2) internal connection (LUT.output -> FF.input nets that can be fully absorbed in a BLE) score
  IndexType numIntNets = computeNumInternalNets(lutA, lutB, ffA, ffB);

  // Compute the total number FFs
  RealType  numFF      = (ffA == kIndexTypeMax ? 0 : 1) + (ffB == kIndexTypeMax ? 0 : 1);

  // Compute the final score
  return 0.1 * numLUTShareInputs + numIntNets - 0.01 * numFF;
}

/// Compute the number of shared inputs of two instances
/// For now, this function is only called for LUT pairs in slot assignment score computation
inline IndexType DLSolver::computeNumShareInputs(const DLInstance &a, const DLInstance &b) const {
  IndexType res  = 0;
  IndexType idxA = findInstInputPinIndex(a, 0);
  IndexType idxB = findInstInputPinIndex(b, 0);
  IndexType netA = _pinArray[a.pinIdArray[idxA]].netId;
  IndexType netB = _pinArray[b.pinIdArray[idxB]].netId;
  while (true) {
    if (netA < netB) {
      if ((idxA = findInstInputPinIndex(a, idxA + 1)) == a.numPins()) {
        return res;
      }
      netA = _pinArray[a.pinIdArray[idxA]].netId;
    } else if (netA > netB) {
      if ((idxB = findInstInputPinIndex(b, idxB + 1)) == b.numPins()) {
        return res;
      }
      netB = _pinArray[b.pinIdArray[idxB]].netId;
    } else {
      // netA == netB
      ++res;
      if ((idxA = findInstInputPinIndex(a, idxA + 1)) == a.numPins() ||
          (idxB = findInstInputPinIndex(b, idxB + 1)) == b.numPins()) {
        return res;
      }
      netA = _pinArray[a.pinIdArray[idxA]].netId;
      netB = _pinArray[b.pinIdArray[idxB]].netId;
    }
  }
}

/// Compute the net sharing score of a cluster (actually LUTs for now)
/// For now, this function is only called for BLEs in slot assignment score computation
/// @param  pins  a list of pins sorted from low to high, which is also the order of the IDs of their incident nets
inline RealType DLSolver::computeNetShareScore(const IndexVector &pins) const {
  if (pins.empty()) {
    return 0;
  }
  const auto *currNet = &_netArray[_pinArray[pins[0]].netId];
  if (currNet->numPins() > _param.netShareScoreMaxNetDegree) {
    return 0;
  }

  // We compute the net sharing score in a net-by-net manner
  RealType  score      = 0;
  IndexType numIntPins = 1;
  IndexType numIntNets = 0;
  IndexType numNets    = 0;
  for (IndexType i = 1; i < pins.size(); ++i) {
    IndexType netId = _pinArray[pins[i]].netId;
    if (netId == currNet->id) {
      // The current pin still share the same net with the previous pin
      ++numIntPins;
    } else {
      // The current pin does not share the same net with the previous pin any more
      // We here compute the net sharing scores for the previous net (currNet)
      ++numNets;
      numIntNets += (numIntPins == currNet->numPins() ? 1 : 0);
      score += currNet->weight * (numIntPins - 1.0) / (currNet->numPins() - 1.0);
      currNet = &_netArray[netId];
      if (currNet->numPins() > _param.netShareScoreMaxNetDegree) {
        break;
      }
      numIntPins = 1;
    }
  }
  // Need to handle the last net as well
  if (currNet->numPins() <= _param.netShareScoreMaxNetDegree) {
    ++numNets;
    numIntNets += (numIntPins == currNet->numPins() ? 1 : 0);
    score += currNet->weight * (numIntPins - 1.0) / (currNet->numPins() - 1.0);
  }
  // Compute the final net sharing score
  score /= (1.0 + _param.extNetCountWt * (numNets - numIntNets));
  return score;
}

/// Compute the number of internal connections in a BLE
/// An internal connection is a LUT.output -> FF.input net
/// For now, this function is only for slot assignment score computation
inline IndexType DLSolver::computeNumInternalNets(IndexType lutA, IndexType lutB, IndexType ffA, IndexType ffB) const {
  IndexType res = 0;
  for (IndexType id : {lutA, lutB}) {
    if (id == kIndexTypeMax) {
      continue;
    }
    // Find the output pin of this LUT
    const auto &lut = _instArray[id];
    auto        rv  = std::find_if(lut.pinIdArray.begin(), lut.pinIdArray.end(), [&](IndexType i) {
      return _pinArray[i].type == DLPinType::OUTPUT;
    });
    if (rv == std::end(lut.pinIdArray)) {
      continue;
    }
    IndexType   outPinId = *rv;
    const auto &outNet   = _netArray[_pinArray[outPinId].netId];
    for (IndexType pinId : outNet.pinIdArray) {
      const auto &pin = _pinArray[pinId];
      if (pin.type == DLPinType::INPUT && (pin.instId == ffA || pin.instId == ffB)) {
        ++res;
      }
    }
  }
  return res;
}

}   // namespace masked_direct_lg
OPENPARF_END_NAMESPACE


#endif   // OPENPARF_OPS_DIRECT_LG_SRC_DL_SOLVER_H_