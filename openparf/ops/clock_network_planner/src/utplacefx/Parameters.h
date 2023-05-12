#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__
#include "util/message.h"
#include "util/namespace.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"
#include "ops/clock_network_planner/src/utplacefx/Vector2D.h"


OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {

/// Class to contain all parameters used in this project
/// All members here are static, so that they can be easilly accessd by any function/object in the project
/// One should not try to create an instance of this class
class Parameters
{
public:
    static RealType               scaledXLen(RealType x)                       { return x * archWirelenScalingX; }
    static RealType               scaledYLen(RealType y)                       { return y * archWirelenScalingY; }
    static XY<RealType>           scaledXYLen(const XY<RealType> &xy)          { return XY<RealType>(xy.x() * archWirelenScalingX, xy.y() * archWirelenScalingY); }
    static void                   scaleXLen(RealType &x)                       { x *= archWirelenScalingX; }
    static void                   scaleYLen(RealType &y)                       { y *= archWirelenScalingY; }
    static void                   scaleXYLen(XY<RealType> &xy)                 { xy.set(xy.x() * archWirelenScalingX, xy.y() * archWirelenScalingY); }

    // Some compile-time constants
    static constexpr IndexType    DYNAMIC_AREA_ADJUSTMENT_OMP_CHUNK_SIZE = 128;
    static constexpr IndexType    CONSENSUS_CLUSTERING_OMP_CHUNK_SIZE = 128;
    static constexpr IndexType    DIRECT_LEGALIZATION_OMP_CHUNK_SIZE = 128;
    static constexpr IndexType    SLOT_ASSIGNMENT_OMP_CHUNK_SIZE = 128;
    static constexpr IndexType    ISM_OMP_CHUNK_SIZE = 8;

    // FPGA architecture parameters
    static bool                   ignoreClockRules;                                            // A switch to ignore all clock constraints
    static IndexType              archClockRegionClockCapacity;                                // Clock region clock capacity
    static IndexType              archHalfColumnRegionClockCapacity;                           // Half-column region clock capacity
    static IndexType              archHalfColumnRegionWidth;                                   // Half-column region width
    static RealType               archWirelenScalingX;                                         // The x-directed wirelength scaling factor
    static RealType               archWirelenScalingY;                                         // The x-directed wirelength scaling factor

    // Parameters for clock network planning
    static IndexType              cnpNetCoarseningMaxNetDegree;                                // The maximum net degree for net coarsening
    static RealType               cnpNetCoarseningMaxClusterDemandSLICE;                       // The maximum resource demand for each SLICE cluster
    static RealType               cnpNetCoarseningMaxClusterDemandDSP;                         // The maximum resource demand for each DSP cluster
    static RealType               cnpNetCoarseningMaxClusterDemandRAM;                         // The maximum resource demand for each RAM cluster
    static RealType               cnpNetCoarseningMaxClusterDimX;                              // The maximum X dimension of the bounding box of a cluster
    static RealType               cnpNetCoarseningMaxClusterDimY;                              // The maximum Y dimension of the bounding box of a cluster
    static IndexType              cnpVDOverflowCost;                                           // The vertical distribution routing overflow cost
    static IndexType              cnpHDOverflowCost;                                           // The horizontal distribution routing overflow cost
    static IndexType              cnpMaxDRouteLRIter;                                          // The maximum number of lagrangian relaxation iterations for distribution layer routing
    static RealType               cnpDRouteDeltaPenaltyTol;                                    // The tolerance for DRoute delta penalty
    static RealType               cnpRRouteOverflowCostIncrValue;                              // The overflow cost increase value for R-layer routing
    static IndexType              cnpMaxRRouteRipupIter;                                       // The maximum ripup and reroute iterations for R-layer routing
    static IndexType              cnpMaxNumLegalSolution;                                      // The maximum number of legal solutions we explore
    static RealType               cnpHalfColumnPlanningGaussianSigma;                          // See function ClockNetworkPlanner::nodeToHcProbability
    static RealType               cnpMinCostFlowCostScaling;                                   // The scaling factor for cost in min-cost flow
    static IndexType              cnpMinCostFlowNodeAssignNumNearestClockRegionInit;           // The initial number of clock regions being considered for each node set during MCF-based assignment
    static IndexType              cnpMinCostFlowNodeAssignNumNearestClockRegionIncr;           // The number of clock regions increased each time for each node set during MCF-based assignment

private:
    // Disable creating instances of this class
    Parameters() {}
};

}
OPENPARF_END_NAMESPACE

#endif // __PARAMETERS_H__
