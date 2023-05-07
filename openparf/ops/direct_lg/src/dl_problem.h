/**
 * File              : dl_problem.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.19.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_DIRECT_LG_SRC_DL_PROBLEM_H_
#define OPENPARF_OPS_DIRECT_LG_SRC_DL_PROBLEM_H_

// C++ standard library headers
#include <limits>
#include <string>
#include <vector>

// project headers
#include "container/vector_2d.hpp"

// local headers
#include "ops/direct_lg/src/direct_lg_db.h"

OPENPARF_BEGIN_NAMESPACE

namespace direct_lg {

using RealType  = float;
using IntType   = int32_t;
using IndexType = uint32_t;
using Byte      = uint8_t;
using container::Box;
using container::Vector2D;
using container::XY;

constexpr IndexType kIndexTypeMax = std::numeric_limits<IndexType>::max();
constexpr RealType  kRealTypeMax  = std::numeric_limits<RealType>::max();
constexpr RealType  kRealTypeMin  = std::numeric_limits<RealType>::lowest();

/// Enum for DL-instance type
enum class DLInstanceType : Byte { LUT, FF, LRAM, SHIFT, DONTCARE };

inline std::string toString(DLInstanceType type) {
  switch (type) {
    case DLInstanceType::LUT:
      return "LUT";
    case DLInstanceType::FF:
      return "FF";
    case DLInstanceType::LRAM:
      return "LRAM";
    case DLInstanceType::SHIFT:
      return "SHIFT";
    case DLInstanceType::DONTCARE:
      return "DONTCARE";
    default:
      openparfAssert(false);
  }
}

/// Enum for DL-pin type
enum class DLPinType : Byte { INPUT, OUTPUT, CK, SR, CE };

/// Enum for DL-site type
enum class DLSiteType : Byte { SLICE, SLICEM, DONTCARE };

inline std::string toString(DLSiteType type) {
  switch (type) {
    case DLSiteType::SLICE:
      return "SLICE";
    case DLSiteType::SLICEM:
      return "SLICEM";
    case DLSiteType::DONTCARE:
      return "DONTCARE";
    default:
      openparfAssert(false);
  }
  return "";
}

/// Class to define a Direct Legalization (DL) problem
struct DLProblem {
  std::vector<XY<RealType>>   instXYs;           // Size = #inst, store initial instance locations
  std::vector<DLInstanceType> instTypes;         // Size = #inst, type of each instance
  std::vector<IndexType>      instDemands;       // Size = #inst, #inputs for LUTs
  std::vector<RealType>       instWts;           // Size = #inst, weight for centroid computation
  std::vector<RealType>       instAreas;         // Size = #inst, area of each instance in GP
  std::vector<IndexType>      pinToInst;         // Size = #pin, instance ID of each pin
  std::vector<IndexType>      pinToNet;          // Size = #pin, net ID of each pin
  std::vector<XY<RealType>>   pinOffsets;        // Size = #pin, pin offset w.r.t. instance pos (maybe
                                                 // center or lower-left corner)
  std::vector<DLPinType>      pinTypes;          // Size = #pin, pin offset w.r.t. instance
                                                 // pos (maybe center or lower-left corner)
  std::vector<RealType>       netWts;            // Size = #net, weight of each net for
                                                 // wirelength optimization
  Vector2D<DLSiteType>        siteTypes;         // Size = #site, type of each site
  Vector2D<XY<RealType>>      siteXYs;           // Size = #site, the actual location of each site
  IndexType                   num_LUTs;          // #LUTs per CLB
  IndexType                   num_FFs;           // #FFs per CLB
  IndexType                   numLUTInst;        // number of LUT instances
  IndexType                   numFFInst;         // number of FF instances
  IndexType                   numSiteSLICE;      // number of SLICE sites.
  std::vector<bool>           isInstFixed;       // Size = #inst, whether the instance is fixed.
  bool                        useXarchLgRule;   // Consider xarch's special constraints. For now it's
                                                 // just LRAM and SHIFT
};
}   // namespace direct_lg

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_DIRECT_LG_SRC_DL_PROBLEM_H_
