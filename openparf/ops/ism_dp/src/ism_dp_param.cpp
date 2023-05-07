/**
 * File              : ism_dp_param.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.14.2021
 * Last Modified Date: 09.14.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ops/ism_dp/src/ism_dp_param.h"

// project headers
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {

ISMDetailedPlaceParam ISMDetailedPlaceParam::ParseFromPyobject(const py::object &pyparam) {
  ISMDetailedPlaceParam param;

  param.routeUtilBinW    = pyparam.attr("routeUtilBinW").cast<decltype(param.routeUtilBinW)>();
  param.routeUtilBinH    = pyparam.attr("routeUtilBinH").cast<decltype(param.routeUtilBinH)>();
  param.routeUtilBinDimX = pyparam.attr("routeUtilBinDimX").cast<decltype(param.routeUtilBinDimX)>();
  param.routeUtilBinDimY = pyparam.attr("routeUtilBinDimY").cast<decltype(param.routeUtilBinDimY)>();
  param.unitHoriRouteCap = pyparam.attr("unitHoriRouteCap").cast<decltype(param.unitHoriRouteCap)>();
  param.unitVertRouteCap = pyparam.attr("unitVertRouteCap").cast<decltype(param.unitVertRouteCap)>();
  param.pinUtilBinW      = pyparam.attr("pinUtilBinW").cast<decltype(param.pinUtilBinW)>();
  param.pinUtilBinH      = pyparam.attr("pinUtilBinH").cast<decltype(param.pinUtilBinH)>();
  param.pinUtilBinDimX   = pyparam.attr("pinUtilBinDimX").cast<decltype(param.pinUtilBinDimX)>();
  param.pinUtilBinDimY   = pyparam.attr("pinUtilBinDimY").cast<decltype(param.pinUtilBinDimY)>();
  param.unitPinCap       = pyparam.attr("unitPinCap").cast<decltype(param.unitPinCap)>();
  param.ffPinWeight      = pyparam.attr("ffPinWeight").cast<decltype(param.ffPinWeight)>();
  param.pinStretchRatio  = pyparam.attr("pinStretchRatio").cast<decltype(param.pinStretchRatio)>();
  param.routeUtilToFixWhiteSpace =
          pyparam.attr("routeUtilToFixWhiteSpace").cast<decltype(param.routeUtilToFixWhiteSpace)>();
  param.pinUtilToFixWhiteSpace = pyparam.attr("pinUtilToFixWhiteSpace").cast<decltype(param.pinUtilToFixWhiteSpace)>();
  param.xWirelenWt             = pyparam.attr("xWirelenWt").cast<decltype(param.xWirelenWt)>();
  param.yWirelenWt             = pyparam.attr("yWirelenWt").cast<decltype(param.yWirelenWt)>();
  param.verbose                = pyparam.attr("verbose").cast<decltype(param.verbose)>();
  param.honorClockConstraints  = pyparam.attr("honorClockConstraints").cast<bool>();
  return param;
}

}   // namespace ism_dp

OPENPARF_END_NAMESPACE
