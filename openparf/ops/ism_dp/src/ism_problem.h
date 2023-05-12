/**
 * File              : ism_problem.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.02.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_ISM_DP_ISM_PROBLEM_H
#define OPENPARF_OPS_ISM_DP_ISM_PROBLEM_H

// C++ standard library headers
#include <array>
#include <functional>
#include <set>
#include <vector>

// project headers
#include "container/vector_2d.hpp"
#include "geometry/geometry.hpp"

// local headers
#include "ops/ism_dp/src/ism_dp_type_trait.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {


using container::Box;
using container::Vector2D;
using container::XY;

/// Class to define a Independent Set Matching (ISM) detailed placement problem
struct ISMProblem {
  using IndexVector = std::vector<IndexType>;
  using CESet       = std::array<IndexType, 2>;
  using IndexSet    = std::set<IndexType>;
  ISMProblem()      = default;

  IndexVector        instTypes;   // Size = #inst, type ID of each instance
  IndexVector        instCKSR;    // Size = #inst, CK/SR ID of each instance
  std::vector<CESet> instCE;      // Size = #inst, CE IDs of each instance
  IndexVector
          instToSite;   // Size = #inst, the initial site ID of each instance. Infinity means this instance is fixed.
  std::vector<XY<RealType>> instXYs;   // Size = #insts, position of fixed instance
  std::vector<IndexVector>
              instToMates;   // Size = #inst, the mate ID of each instance that should be given extra credit
  IndexVector pinToInst;     // Size = #pin, instance ID of each pin
  IndexVector pinToNet;      // Size = #pin, net ID of each pin
  std::vector<XY<RealType>> pinOffsets;      // Size = #pin, pin offset w.r.t. instance lower-left corner
  std::vector<RealType>     netWts;          // Size = #net, weight of each net for wirelength optimization
  Vector2D<XY<RealType>>    siteXYs;         // Size = #site, the actual location of each site
  Vector2D<IndexType>       siteToCtrlSet;   // Size = #site, the control set ID of each site
  Vector2D<IndexType>       siteToMate;      // Size = #site, the mate ID of each site
  std::vector<IndexSet>     instCKs;         // Size = #inst, number of clocks connected to each instance
  std::function<bool(IndexType, IndexType, IndexType)> isCKAllowedInSite;   // func(clockId, siteX, siteY)
  bool                                                 honorClockConstraints;
};

}   // namespace ism_dp
OPENPARF_END_NAMESPACE

#endif
