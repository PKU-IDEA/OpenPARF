#include "util/namespace.h"

#include "ops/clock_network_planner/src/utplacefx/Types.h"
#include "ops/clock_network_planner/src/utplacefx/Parameters.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {

// FPGA architecture parameters
bool                   Parameters::ignoreClockRules                                            = false;  // A switch to ignore all clock constraints
IndexType              Parameters::archClockRegionClockCapacity                                = 24;     // Clock region clock capacity
IndexType              Parameters::archHalfColumnRegionClockCapacity                           = 12;     // Half-column region clock capacity
IndexType              Parameters::archHalfColumnRegionWidth                                   = 2;      // Half-column region width
RealType               Parameters::archWirelenScalingX                                         = 0.7;    // The x-directed wirelength scaling factor
RealType               Parameters::archWirelenScalingY                                         = 1.2;    // The x-directed wirelength scaling factor

// Parameters for clock network planning
IndexType              Parameters::cnpNetCoarseningMaxNetDegree                                = 16;     // The maximum net degree for net coarsening
RealType               Parameters::cnpNetCoarseningMaxClusterDemandSLICE                       = 100.0;  // The maximum resource demand for each SLICE cluster
RealType               Parameters::cnpNetCoarseningMaxClusterDemandDSP                         = 2.0;    // The maximum resource demand for each DSP cluster
RealType               Parameters::cnpNetCoarseningMaxClusterDemandRAM                         = 2.0;    // The maximum resource demand for each RAM cluster
RealType               Parameters::cnpNetCoarseningMaxClusterDimX                              = 5.0;    // The maximum X dimension of the bounding box of a cluster
RealType               Parameters::cnpNetCoarseningMaxClusterDimY                              = 5.0;    // The maximum Y dimension of the bounding box of a cluster
IndexType              Parameters::cnpVDOverflowCost                                           = 1;      // The vertical distribution routing overflow cost
IndexType              Parameters::cnpHDOverflowCost                                           = 1;      // The horizontal distribution routing overflow cost
IndexType              Parameters::cnpMaxDRouteLRIter                                          = 20;     // The maximum number of lagrangian relaxation iterations for distribution layer routing
RealType               Parameters::cnpDRouteDeltaPenaltyTol                                    = 1e-6;   // The tolerance for DRoute delta penalty
RealType               Parameters::cnpRRouteOverflowCostIncrValue                              = 1.0;    // The overflow cost increase value for R-layer routing
IndexType              Parameters::cnpMaxRRouteRipupIter                                       = 100;    // The maximum ripup and reroute iterations for R-layer routing
IndexType              Parameters::cnpMaxNumLegalSolution                                      = 1;      // The maximum number of legal solutions we explore
RealType               Parameters::cnpHalfColumnPlanningGaussianSigma                          = 2.0;    // See function ClockNetworkPlanner::nodeToHcProbability
RealType               Parameters::cnpMinCostFlowCostScaling                                   = 1000.0; // The scaling factor for cost in min-cost flow
IndexType              Parameters::cnpMinCostFlowNodeAssignNumNearestClockRegionInit           = 4;      // The initial number of clock regions being considered for each node set during MCF-based assignment
IndexType              Parameters::cnpMinCostFlowNodeAssignNumNearestClockRegionIncr           = 4;      // The number of clock regions increased each time for each node set during MCF-based assignment
}
OPENPARF_END_NAMESPACE

