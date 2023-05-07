/**
 * File              : ism_param.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.02.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_ISM_DP_ISM_PARAM_H
#define OPENPARF_OPS_ISM_DP_ISM_PARAM_H

// C++ standard library headers
#include <string>
#include <vector>

// local headers
#include "ops/ism_dp/src/ism_dp_type_trait.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

/// Defines parameters needed by ISMSolver
struct ISMParam {
  IndexType maxNetDegree      = 16;      // We ignore any nets larger than this value for wirelength computation
  IndexType numSitePerCtrlSet = 1;       // Number of sites in each site control set
  IndexType maxRadius         = 10;      // Maximum independent set radius from the seed
  IndexType siteXStep         = 1;       // The x step when grow an independent set in a spiral access
  IndexType siteYInterval     = 1;       // The y interval when grow an independent set in a spiral access
  IndexType maxIndepSetSize   = 50;      // The maximum independent set size
  RealType  mateCredit        = 0.0;     // The extra credit given to instance-to-mate site assignment
  RealType  flowCostScale     = 100.0;   // Flow costs must be integer, we up scale them to get better accuracy
  IndexType batchSize         = 10;      // We check stop condition every this number of ISM iterations
  RealType  minBatchImprov = 0.001;   // Stop ISM if the wirelength improv. is less than this among for a single batch
  RealType  xWirelenWt     = 1.0;     // The weight for x-directed wirelength
  RealType  yWirelenWt     = 1.0;     // The weight for y-directed wirelength

  // For message printing
  // 0: quiet
  // 1: basic messages
  // 2: extra messages
  IndexType verbose        = 1;
};

}   // namespace ism_dp

OPENPARF_END_NAMESPACE

#endif
