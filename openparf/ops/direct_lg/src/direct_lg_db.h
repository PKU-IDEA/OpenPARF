/**
 * File              : direct_lg_db.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 06.16.2020
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_DB_H_
#define OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_DB_H_

// C++ standard library headers
#include <memory>
// The project's .h files
#include "database/clock_availability.h"
#include "database/placedb.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace direct_lg {

template<typename T>
struct DirectLegalizeDB {
  int32_t const *pin2inst            = nullptr;   ///< #pins
  int32_t const *pin2net             = nullptr;   ///< #pins
  uint8_t const *pin_signal_directs  = nullptr;   ///< #pins
  uint8_t const *pin_signal_types    = nullptr;   ///< #pins
  uint8_t const *is_inst_luts        = nullptr;   ///< #insts
  uint8_t const *is_inst_ffs         = nullptr;   ///< #insts
  T const *      inst_sizes          = nullptr;   ///< #insts x 2
  T const *      pin_offsets         = nullptr;   ///< #pins x 2
  T const *      site_bboxes         = nullptr;   ///< #sites x 4
  T const *      net_weights         = nullptr;   ///< #nets
  int32_t const *site_lut_capacities = nullptr;   ///< #sites
  int32_t const *site_ff_capacities  = nullptr;   ///< #sites
  int32_t        num_insts;                       ///< number of instances
  int32_t        num_nets;                        ///< number of nets
  int32_t        num_pins;                        ///< number of pins
  int32_t        num_sites;                       ///< number of sites
  int32_t        site_map_dim[2];                 ///< dimension of site map
};


/**
 * @brief Defines parameters needed by `DLSolver`.
 *
 */
struct DirectLegalizeParam {
  static DirectLegalizeParam ParseFromPyObject(const py::object &pyparam);

  uint32_t                   netShareScoreMaxNetDegree;   // We ignore any nets larger than this
                                                          // value for net sharing score
                                                          // computation
  uint32_t                   wirelenScoreMaxNetDegree;    // We ignore any nets larger than this
                                                          // value for wirelength score computation
  double                     preclusteringMaxDist;        // The max distance in th preclustering phase
  double                     nbrDistBeg;                  // The initial site neighbor distance
  double                     nbrDistEnd;                  // The maximum site neighbor distance
  double                     nbrDistIncr;                 // The distance step of adjacent neighbor groups
  uint32_t                   candPQSize;                  // The candidate PQ size
  uint32_t                   minStableIter;               // Minimum number of stable iterations before a top
                                                          // candidate can be committed
  uint32_t                   minNeighbors;                // The minimum number of active instances neighbors
                                                          // for each site
  double                     extNetCountWt;               // The coefficient for the number of external net
                                                          // count in the candidate score computation
  double                     wirelenImprovWt;             // The coefficient for the wirelength improvement in
                                                          // the candidate score computation
  uint32_t                   greedyExpansion;             // The extra radius value to search after the
                                                          // first legal location in greedy legalization
  uint32_t                   ripupExpansion;              // The extra radius value to search after the first
                                                          // legal location in ripup legalization
  double                     slotAssignFlowWeightScale;   // Flow weights must be integer, we up
                                                          // scale them to get better accuracy
  double                     slotAssignFlowWeightIncr;    // The flow weight increase step size for
                                                          // max-weight matching based LUT pairing
  double                     xWirelenWt;                  // The weight for x-directed wirelength
  double                     yWirelenWt;                  // The weight for y-directed wirelength

  // For message printing
  // 0: quiet
  // 1: basic messages
  // 2: extra messages
  uint32_t                   verbose;

  uint32_t                   CLB_capacity;             // Number of LUT/FF slots in each slice
  uint32_t                   BLE_capacity;             // Number of LUT/FF slots in each BLE
  uint32_t                   half_CLB_capacity;        // Half number of LUT/FF slots in each slice
  uint32_t                   num_BLEs_per_CLB;         // Number of BLEs per slice
  uint32_t                   num_BLEs_per_half_CLB;    // Number of BLEs per half slice
  uint32_t                   omp_dynamic_chunk_size;   // The chunk size for OpenMP dynamic
                                                       // scheduling

  // Clock related information
  bool                       honorClockRegionConstraint;
  bool                       honorHalfColumnConstraint;
  uint32_t                   numClockNet;
  uint32_t                   numHalfColumn;
  uint32_t                   maxClockNetPerHalfColumn;

  // Whether use xarch benchmark.
  bool                       useXarchLgRule;
};


}   // namespace direct_lg

OPENPARF_END_NAMESPACE

#endif  // OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_DB_H_
