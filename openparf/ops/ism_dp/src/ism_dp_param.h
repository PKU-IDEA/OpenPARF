/**
 * File              : ism_dp_param.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.02.2020
 * Last Modified Date: 07.14.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_ISM_DP_SRC_ISM_DP_PARAM_H_
#define OPENPARF_OPS_ISM_DP_SRC_ISM_DP_PARAM_H_

// C++ standard library headers
#include <cstdint>

// project headers
#include "util/namespace.h"
#include "util/torch.h"

OPENPARF_BEGIN_NAMESPACE

namespace ism_dp {
/// This class defines parameters needed by DetailedPlacer
struct ISMDetailedPlaceParam {
  static ISMDetailedPlaceParam ParseFromPyobject(const py::object &pyparam);

  double                       routeUtilBinW;      ///< The bin width for routing estimation
  double                       routeUtilBinH;      ///< The bin height for routing estimation
  uint32_t                     routeUtilBinDimX;   ///< The bin dimension for routing estimation
  uint32_t                     routeUtilBinDimY;   ///< The bin dimension for routing estimation
  double                       unitHoriRouteCap;   ///< The unit horizontal routing capacity
  double                       unitVertRouteCap;   ///< The unit vertical routing capacity
  double                       pinUtilBinW;        ///< The bin width for pin utilization estimation
  double                       pinUtilBinH;        ///< The bin height for pin utilization estimation
  uint32_t                     pinUtilBinDimX;     ///< The bin dimension for pin utilization estimation
  uint32_t                     pinUtilBinDimY;     ///< The bin dimension for pin utilization estimation
  double                       unitPinCap;         ///< The unit pin capacity
  double                       ffPinWeight;        ///< The weight of FF pins for pin density optimization
  double   pinStretchRatio;            ///< Stretch pin to a ratio of the pin utilization bin to smooth the density map
  double   routeUtilToFixWhiteSpace;   ///< We fix white spaces at places where routing utilization is higher than this
  double   pinUtilToFixWhiteSpace;     ///< We fix white spaces at places where pin utilization is higher than this
  double   xWirelenWt;                 ///< weight for wirelength in x direction
  double   yWirelenWt;                 ///< weight for wirelength in y direction
  int32_t  verbose;                    ///< Verbose flag
  bool     honorClockConstraints;      ///< Whether honor clock region constraints.
  uint8_t *fixedMask;   ///< Besides SSMIR instances like IOs, instances marked fixed will not be moved during detailed
                        ///< placement. Ensure that this will not affect other instances.
};

}   // namespace ism_dp


OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_ISM_DP_SRC_ISM_DP_PARAM_H_
