/**
 * File              : direct_lg_kernel.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.05.2021
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_KERNEL_H_
#define OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_KERNEL_H_


// C++ standard library headers
#include <vector>

// project headers
#include "database/clock_availability.h"
#include "database/placedb.h"
#include "util/namespace.h"

// local headers
#include "ops/direct_lg/src/direct_lg_db.h"
#include "ops/direct_lg/src/dl_problem.h"
#include "ops/direct_lg/src/dl_solver.h"

OPENPARF_BEGIN_NAMESPACE

namespace direct_lg {

void initDLProblem(database::PlaceDB const &db, DLProblem &prob);
void writeHalfColumnAvailabilityMapSolution(database::PlaceDB const &db, DLSolver const &solver, uint8_t *hcAvailMap);

/// @brief write DL solution back to the netlist
template<typename T>
void writeDLSolution(database::PlaceDB const &db,
                     DLProblem const &        prob,
                     DLSolver const &         solver,
                     T const *                init_pos,
                     T *                      pos) {
  for (uint32_t i = 0; i < db.numInsts(); ++i) {
    if (db.isInstLUT(i) || db.isInstFF(i)) {
      // Note that we only move LUT/FF in DL
      // So we only need to commit their solution from our DL solver
      const auto &sol = solver.instSol(i);
      const auto &loc = prob.siteXYs[sol.siteId];
      pos[i * 3]      = loc.x();
      pos[i * 3 + 1]  = loc.y();
      pos[i * 3 + 2]  = sol.z;
    } else {
      pos[i * 3]     = init_pos[i * 2];
      pos[i * 3 + 1] = init_pos[i * 2 + 1];
    }
  }
}

template<typename T>
void directLegalize(database::PlaceDB const &                db,
                    DirectLegalizeParam const &              param,
                    T const *                                init_pos,
                    ClockAvailCheckerType<T>                 legality_check_functor,
                    LayoutXy2GridIndexFunctorType<T>         xy_to_half_column_functor,
                    const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
                    int32_t                                  num_threads,
                    T *                                      pos,
                    uint8_t *                                hc_avail_map) {
  DLProblem prob;
  prob.useXarchLgRule = param.useXarchLgRule;
  prob.instXYs.resize(db.numInsts());
  for (IndexType i = 0; i < prob.instXYs.size(); ++i) {
    prob.instXYs[i].set(init_pos[(i << 1)], init_pos[(i << 1) + 1]);
  }
  initDLProblem(db, prob);

  // Perform the DL
  DLSolver solver(prob, param, legality_check_functor, xy_to_half_column_functor, inst_to_clock_indexes, num_threads);
  solver.run();

  // Write the DL solution back to the netlist
  writeDLSolution(db, prob, solver, init_pos, pos);
  openparfPrint(kDebug, "writeDLSolution done.\n");

  // Return the half column solution
  if (param.honorHalfColumnConstraint) {
    writeHalfColumnAvailabilityMapSolution(db, solver, hc_avail_map);
  }

  if (false) {
    // debug
    auto const &design  = db.db()->design();
    auto const &layout  = db.db()->layout();
    auto const &netlist = design.netlist();
    for (int i = 0; i < netlist->numInsts(); i++) {
      if (db.isInstClockSource(i)) {
        continue;
      }
      double xx = pos[i * 3];
      double yy = pos[i * 3 + 1];
      if (!legality_check_functor(i, xx, yy)) {
        int         inst_id = db.oldInstId(i);
        auto const &inst    = netlist->inst(inst_id);
        openparfPrint(kDebug,
                      "new_id: %d, old_id: %d, name: %s, (%lf, %lf)->(%lf, %lf)\n",
                      i,
                      inst_id,
                      inst.attr().name().c_str(),
                      init_pos[i << 1],
                      init_pos[i << 1 | 1],
                      xx,
                      yy);
      }
    }
  }
}

/// @brief It is necesssary to add another wrapper with the noinline attribute.
/// This is a compiler related behavior where GCC will generate symbols with
/// "._omp_fn" suffix for OpenMP functions, causing the mismatch between
/// function declaration and implementation. As a result, at runtime, the
/// symbols will be not found. As GCC may also automatically inline the
/// function, we need to add this guard to guarantee noinline.
template<typename T>
void OPENPARF_NOINLINE directLegalizeLauncher(database::PlaceDB const &                db,
                                              DirectLegalizeParam const &              param,
                                              T const *                                init_pos,
                                              ClockAvailCheckerType<T>                 legality_check_functor,
                                              LayoutXy2GridIndexFunctorType<T>         xy_to_half_column_functor,
                                              const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
                                              int32_t                                  num_threads,
                                              T *                                      pos,
                                              uint8_t *                                hc_avail_map) {
  directLegalize(db,
                 param,
                 init_pos,
                 legality_check_functor,
                 xy_to_half_column_functor,
                 inst_to_clock_indexes,
                 num_threads,
                 pos,
                 hc_avail_map);
}

}   // namespace direct_lg

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_KERNEL_H_
