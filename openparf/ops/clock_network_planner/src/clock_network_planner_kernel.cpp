/**
 * @file   clock_network_planner_kernel.cpp
 * @author Yibai Meng
 * @date   Sep 2020
 */
#include <iostream>

#include "database/placedb.h"
#include "ops/clock_network_planner/src/database_wrapper.h"
#include "ops/clock_network_planner/src/netlist_wrapper.h"
#include "ops/clock_network_planner/src/utplacefx/ClockNetworkPlanner.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"
#include "ops/clock_network_planner/src/utplacefx/Parameters.h"

OPENPARF_BEGIN_NAMESPACE

// TODO: using static pointers seems to be a bad way to do stuff
static utplacefx::ClockNetworkPlanner<double> *cnp_double        = nullptr;
static utplacefx::ClockNetworkPlanner<float> * cnp_float         = nullptr;
static utplacefx::WrapperNetlist<double> *     wrapper_nl_double = nullptr;
static utplacefx::WrapperNetlist<float> *      wrapper_nl_float  = nullptr;

template<typename T>
static void OPENPARF_NOINLINE cnpInitLauncher(database::PlaceDB const &placedb) {
    openparfAssertMsg(false, "Invalid type, only float and double are supported");
}
// TODO: pass the type information into CNP?
template<>
void OPENPARF_NOINLINE cnpInitLauncher<float>(database::PlaceDB const &placedb) {
    auto db          = placedb.db();
    wrapper_nl_float = new utplacefx::WrapperNetlist<float>(placedb);
    auto wrapper_db  = new utplacefx::WrapperDatabase(db->layout());
    cnp_float        = new utplacefx::ClockNetworkPlanner<float>(wrapper_db);
    cnp_float->setNetlist(wrapper_nl_float);
}

template<>
void OPENPARF_NOINLINE cnpInitLauncher<double>(database::PlaceDB const &placedb) {
    auto db           = placedb.db();
    wrapper_nl_double = new utplacefx::WrapperNetlist<double>(placedb);
    auto wrapper_db   = new utplacefx::WrapperDatabase(db->layout());
    cnp_double        = new utplacefx::ClockNetworkPlanner<double>(wrapper_db);
    cnp_double->setNetlist(wrapper_nl_double);
}

template<typename T>
static void runClockNetworkAssignment(T const *pos, int32_t num_threads, int32_t *nodeToCr,
                                      uint8_t *clkAvailCR, T const *inflated_areas) {
    openparfAssertMsg(false, "Invalid type, only float and double are supported");
}

template<>
void runClockNetworkAssignment<float>(float const *pos, int32_t num_threads, int32_t *nodeToCr,
                                      uint8_t *clkAvailCR, float const *inflated_areas) {
    wrapper_nl_float->setPos(pos);
    wrapper_nl_float->setInflatedAreas(inflated_areas);
    cnp_float->run();
    cnp_float->transferSolutionToTorchTensor(nodeToCr, clkAvailCR);
}

template<>
void runClockNetworkAssignment<double>(double const *pos, int32_t num_threads, int32_t *nodeToCr,
                                       uint8_t *clkAvailCR, double const *inflated_areas) {
    wrapper_nl_double->setPos(pos);
    wrapper_nl_double->setInflatedAreas(inflated_areas);
    cnp_double->run();
    cnp_double->transferSolutionToTorchTensor(nodeToCr, clkAvailCR);
}


// See comment on openparf/ops/direct_lg/src/direct_lg_kernel.cpp for
// the purpose of this wrapper
template<typename T>
void OPENPARF_NOINLINE clockNetworkPlannerLauncher(T const *pos, int32_t num_threads,
                                                   int32_t *nodeToCr, uint8_t *clkAvailCR,
                                                   T const *inflated_areas) {
    runClockNetworkAssignment(pos, num_threads, nodeToCr, clkAvailCR, inflated_areas);
}

OPENPARF_END_NAMESPACE