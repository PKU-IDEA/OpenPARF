/**
 * File              : direct_lg_db.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.05.2021
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ops/masked_direct_lg/src/direct_lg_db.h"

OPENPARF_BEGIN_NAMESPACE

namespace masked_direct_lg {

DirectLegalizeParam DirectLegalizeParam::ParseFromPyObject(const py::object &pyparam) {
  DirectLegalizeParam param;

  param.netShareScoreMaxNetDegree =
          pyparam.attr("netShareScoreMaxNetDegree").cast<decltype(param.netShareScoreMaxNetDegree)>();
  param.wirelenScoreMaxNetDegree =
          pyparam.attr("wirelenScoreMaxNetDegree").cast<decltype(param.wirelenScoreMaxNetDegree)>();
  param.preclusteringMaxDist = pyparam.attr("preclusteringMaxDist").cast<decltype(param.preclusteringMaxDist)>();
  param.nbrDistBeg           = pyparam.attr("nbrDistBeg").cast<decltype(param.nbrDistBeg)>();
  param.nbrDistEnd           = pyparam.attr("nbrDistEnd").cast<decltype(param.nbrDistEnd)>();
  param.nbrDistIncr          = pyparam.attr("nbrDistIncr").cast<decltype(param.nbrDistIncr)>();
  param.candPQSize           = pyparam.attr("candPQSize").cast<decltype(param.candPQSize)>();
  param.minStableIter        = pyparam.attr("minStableIter").cast<decltype(param.minStableIter)>();
  param.minNeighbors         = pyparam.attr("minNeighbors").cast<decltype(param.minNeighbors)>();
  param.extNetCountWt        = pyparam.attr("extNetCountWt").cast<decltype(param.extNetCountWt)>();
  param.wirelenImprovWt      = pyparam.attr("wirelenImprovWt").cast<decltype(param.wirelenImprovWt)>();
  param.greedyExpansion      = pyparam.attr("greedyExpansion").cast<decltype(param.greedyExpansion)>();
  param.ripupExpansion       = pyparam.attr("ripupExpansion").cast<decltype(param.ripupExpansion)>();
  param.slotAssignFlowWeightScale =
          pyparam.attr("slotAssignFlowWeightScale").cast<decltype(param.slotAssignFlowWeightScale)>();
  param.slotAssignFlowWeightIncr =
          pyparam.attr("slotAssignFlowWeightIncr").cast<decltype(param.slotAssignFlowWeightIncr)>();
  param.xWirelenWt             = pyparam.attr("xWirelenWt").cast<decltype(param.xWirelenWt)>();
  param.yWirelenWt             = pyparam.attr("yWirelenWt").cast<decltype(param.yWirelenWt)>();
  param.verbose                = pyparam.attr("verbose").cast<decltype(param.verbose)>();
  param.CLB_capacity           = pyparam.attr("CLB_capacity").cast<decltype(param.CLB_capacity)>();
  param.BLE_capacity           = pyparam.attr("BLE_capacity").cast<decltype(param.BLE_capacity)>();
  param.half_CLB_capacity      = param.CLB_capacity / 2;
  param.num_BLEs_per_CLB       = param.CLB_capacity / param.BLE_capacity;
  param.num_BLEs_per_half_CLB  = param.half_CLB_capacity / param.BLE_capacity;
  param.omp_dynamic_chunk_size = pyparam.attr("omp_dynamic_chunk_size").cast<decltype(param.omp_dynamic_chunk_size)>();
  param.numClockNet            = pyparam.attr("numClockNet").cast<decltype(param.numClockNet)>();
  param.numHalfColumn          = pyparam.attr("numHalfColumn").cast<decltype(param.numHalfColumn)>();
  param.maxClockNetPerHalfColumn =
          pyparam.attr("maxClockNetPerHalfColumn").cast<decltype(param.maxClockNetPerHalfColumn)>();
  param.useXarchLgRule = pyparam.attr("useXarchLgRule").cast<decltype(param.useXarchLgRule)>();
  return param;
}

}   // namespace masked_direct_lg

OPENPARF_END_NAMESPACE
