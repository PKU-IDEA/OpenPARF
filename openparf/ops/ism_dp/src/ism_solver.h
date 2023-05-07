/**
 * File              : ism_solver.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.02.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_SOLVER_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_SOLVER_H_

// C++ standard library headers
#include <array>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

// 3rdparty headers
#include "boost/container/flat_map.hpp"
#include "lemon/list_graph.h"
#include "lemon/network_simplex.h"

// project headers
#include "container/spiral_accessor.hpp"

// local headers
#include "ops/ism_dp/src/ism_dp_type_trait.h"
#include "ops/ism_dp/src/ism_param.h"
#include "ops/ism_dp/src/ism_problem.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

using container::SpiralAccessor;

/// The kernel to solve independent set matching (ISM) based detailed placement problem
class ISMSolver {
 private:
  static constexpr IndexType OMP_DYNAMIC_CHUNK_SIZE = 8;   // The chunk size for OpenMP dynamic scheduling

  using FlowIntType                                 = std::int64_t;
  using ByteVector                                  = std::vector<Byte>;
  using IndexVector                                 = std::vector<IndexType>;
  using IndexSet                                    = std::set<IndexType>;
  using CESet                                       = std::array<IndexType, 2>;

 public:
  /// Class to represent the final DL solution of an instance
  struct ISMInstanceSol {
    ISMInstanceSol() = default;
    void set(IndexType a, bool b) {
      openparfAssert(!isFixed());
      siteId = a;
      ceFlip = b;
    }

    bool         isFixed() const { return siteId == kIndexTypeMax; }
    IndexType    siteId = kIndexTypeMax;   // ID of the assigned site. If nifinity, it means this ISM instance is Fixed.
    bool         ceFlip = false;           // Whether the CEs are flipped
    XY<RealType> fixedSol;                 // Fixed ISM instance position.
  };

 private:
  /// Class for instances used in ISM
  struct ISMInstance {
    IndexType numPins() const { return pinIdArray.size(); }

    IndexType origId = kIndexTypeMax;   // The original index of this instance when it is constructed
    IndexType id     = kIndexTypeMax;   // Instance ID
    IndexType type = kIndexTypeMax;   // Only instances with the same type can be swapped. Infinity means this instances
                                      // can not be move.
    IndexType cksr = kIndexTypeMax;   // ID of CK/SR for FF
    CESet     ce   = {{kIndexTypeMax, kIndexTypeMax}};   // ID of CE for FF
    IndexSet  ckArray;                                   // IDs of CK nets connected to this inst
    IndexVector    pinIdArray;                           // IDs of incident pins of this instance
    IndexVector    mates;         // Get extra credit if this instance and its mates are assigned to two mate sites
    IndexVector    conn;          // Instances that connects to this instances
    IndexType      numMovs = 0;   // Number of moves of this instance
    ISMInstanceSol sol;           // The final solution after ISM, contains (site ID, ceFlip)
  };

  /// Class for nets used in ISM
  struct ISMNet {
    IndexType     numPins() const { return pinIdArray.size(); }

    IndexVector   pinIdArray;   // IDs of pins in this net
    RealType      weight = 0;   // Wirelength weight
    Box<RealType> bbox;         // Net bounding box
  };

  /// Class for pins used in ISM
  struct ISMPin {
    RealType     offsetX() const { return offset.x(); }
    RealType     offsetY() const { return offset.y(); }

    IndexType    instId = kIndexTypeMax;   // ID of the incident instance
    IndexType    netId  = kIndexTypeMax;   // ID of the incident net
    XY<RealType> offset;                   // Pin offset w.r.t. instance
  };

  /// Class for sites used in ISM
  struct ISMSite {
    RealType     x() const { return loc.x(); }
    RealType     y() const { return loc.y(); }

    XY<RealType> loc;                         // Site location
    IndexType    instId    = kIndexTypeMax;   // ID of the instance in this site
    IndexType    ctrlSetId = kIndexTypeMax;   // ID of the control set this site belongs to
    IndexType    mateId    = kIndexTypeMax;   // The mate site of this site
  };

  /// Class for control sets used in ISM
  struct ISMCtrlSet {
    IndexType cksr      = kIndexTypeMax;                      // ID of CK/SR
    IndexType cksrCount = 0;                                  // Number of instances in this control set has the CK/SR
    CESet     ce        = {{kIndexTypeMax, kIndexTypeMax}};   // ID of CEs
    CESet     ceCount   = {{0, 0}};                           // Number of instancse in this control set has the CEs
  };

  /// Class for building independent set
  struct IndepSet {
    IndexType   type;   // Instance type of this independent set
    IndexVector set;    // Instances in the current independent set
    IndexType   cksr;   // ID of the CK/SR of this independent set
    CESet       ce;     // ID of the CEs of this independent set
    ByteVector  dep;    // Flags to indicate whether an instance is dependent to the current instance set
  };

  /// Class for memory used by an ISM solving
  struct ISMMemory {
    ISMMemory() : graphPtr(new lemon::ListDigraph()) {}

    // For building ISM
    std::vector<Box<RealType>> bboxes;   // Bounding boxes of nets that are incident to the instances in the set
    std::vector<XY<RealType>> offset;   // offset[i] is the pin offset for the insatnce in net bounding box bboxArray[i]
    IndexVector               netIds;   // netIds[i] is the ID of the net corresponding to bboxArray[i]
    IndexVector
            ranges;   // [ranges[i], ranges[i+1]) contains the net bounding box related to the i-th instance in the set
    Vector2D<FlowIntType>                        costMtx;   // Cost matrix

    std::vector<bool>                            canConsider;   // If the ith element can't be swapped with ANY
    // For solving ISM
    // Since lemon::ListDigraph is not constructible, we need to wrap it with std::unique_ptr
    std::unique_ptr<lemon::ListDigraph>          graphPtr;
    std::vector<lemon::ListDigraph::Node>        lNodes;
    std::vector<lemon::ListDigraph::Node>        rNodes;
    std::vector<lemon::ListDigraph::Arc>         lArcs;
    std::vector<lemon::ListDigraph::Arc>         rArcs;
    std::vector<lemon::ListDigraph::Arc>         mArcs;
    std::vector<std::pair<IndexType, IndexType>> mArcPairs;
    IndexVector sol;   // Matching solution, the i-th instance is moved to sol[i]-th instance's location

    // General purpose buffers
    IndexVector idxVec;
  };

 public:
  ISMSolver(const ISMProblem &prob, const ISMParam &param, IndexType num_threads, bool honorClockConstraints = false);

  void                  buildISMProblem(const ISMProblem &prob);
  void                  run();
  XY<RealType>          getLoc(IndexType inst_id) const;

  // Get ISM final solution of a instnace
  const ISMInstanceSol &instSol(IndexType i) const { return _instArray[_instIdMapping[i]].sol; }

 private:
  // Shortcut getters
  IndexType    numInsts() const { return _instArray.size(); }
  IndexType    numNets() const { return _netArray.size(); }
  IndexType    numPins() const { return _pinArray.size(); }
  IndexType    numSites() const { return _siteMap.size(); }

  // Build ISM problem
  void         buildNetlist(const ISMProblem &prob);
  void         buildSiteMap(const ISMProblem &prob);
  void         sortNetlist();
  void         sortNets(std::function<bool(IndexType, IndexType)> cmp);
  void         sortPins(std::function<bool(IndexType, IndexType)> cmp);
  void         sortInstances(std::function<bool(IndexType, IndexType)> cmp);
  void         removeNetlistRedundancy();
  void         removeRedundantPins();
  void         removeRedundantNets();
  void         applyPinIdMapping(const IndexVector &pinIdMapping);

  // ISM initialization
  void         initISM();
  void         computeAllNetBBoxes();
  void         initInstConn();

  // ISM kernel functions
  void         computeInstancePriority();
  void         buildIndependentIndepSets(IndepSet &indepSet, std::vector<IndexVector> &sets);
  void         buildIndepSet(const ISMInstance &seed, IndepSet &indepSet);
  IndexType    instToIndepSetFeasibility(const ISMInstance &inst, const IndepSet &indepSet);
  IndexType    computeCEMergeCost(IndexType a0, IndexType a1, IndexType b0, IndexType b1) const;
  void         addInstToIndepSet(const ISMInstance &inst, IndexType ceFlip, IndepSet &indepSet);
  void         computeNetBBoxes(const IndexVector &set, ISMMemory &mem) const;
  void         computeCostMatrix(const IndexVector &set, ISMMemory &mem) const;
  void         computeMatching(ISMMemory &mem) const;
  void         realizeMatching(const IndexVector &set, ISMMemory &mem);
  bool         stopCondition();
  XY<RealType> computeHPWL(IndexType maxNetDegree = kIndexTypeMax) const;

  // Helping functions
  IndexType    floorAlign(IndexType v, IndexType a) const { return v / a * a; }

 private:
  const ISMParam &         _param;
  SpiralAccessor           _spiralAccessor;   // The spiral accessor to help add instance/site spatial neighbors
  Box<IndexType>           _bndBox;           // The bounding box of the placement region, helps spiral accessing

  std::vector<ISMInstance> _instArray;       // All instances
  std::vector<ISMNet>      _netArray;        // All nets
  std::vector<ISMPin>      _pinArray;        // All pins
  Vector2D<ISMSite>        _siteMap;         // The site map
  std::vector<ISMCtrlSet>  _ctrlSetArray;    // All control sets for sites
  IndexVector              _instIdMapping;   // Original instance ID to sorted instance ID mapping
  IndexType                _numPhyInsts;     // Number of instances that have pins

  std::vector<IndexVector> _movBuckets;       // For bucket sort on instance #moves
  IndexVector              _priority;         // Instance ISM priority
  IndepSet                 _indepSet;         // The helper object for independent set building
  std::vector<IndexVector> _sets;             // Independent "independent sets"
  std::vector<ISMMemory>   _threadMemArray;   // Memory used by each thread

  std::function<bool(IndexType, IndexType, IndexType)> _isCKAllowedInSite;   // func(clockId, siteX, siteY)
  bool                                                 _honorClockConstraints;
  IndexType                                            _iter;          // Number of ISM iterations done
  XY<RealType>                                         _wl;            // Current unscaled x- and y-derected HPWL
  IndexType                                            _num_threads;   // number of threads
};

}   // namespace ism_dp
OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_SOLVER_H_
