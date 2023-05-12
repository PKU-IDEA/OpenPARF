/**
 * File              : direct_lg.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.05.2021
 * Last Modified Date: 09.05.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */

#ifndef OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_H_
#define OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_H_

// C++ standard library headers
#include <tuple>
#include <vector>

// project headers
#include "database/placedb.h"
#include "util/torch.h"

OPENPARF_BEGIN_NAMESPACE

/**
 * @brief Directly legalizes the LUT & FF instances without clock constraint for
 * ISPD benchmark, and LUT5/6, LRAM, SHIFT & DFF for Xarch benchmark.
 * @param placedb Placement database
 * @param pyparam Python object recording the parameters
 * @param init_pos  Inital instance position.
 * @return at::Tensor Post-legalization instance position.
 */
at::Tensor directLegalizeForward(database::PlaceDB const &placedb, py::object pyparam, at::Tensor init_pos);

/**
 * @brief Directly legalizes the LUT & FF instances with clock constraint for
 * ISPD benchmark.
 * @param placedb Placement database
 * @param pyparam Python object recording the parameters
 * @param init_pos  Inital instance position.
 * @return at::Tensor Post-legalization instance position.
 */
std::tuple<at::Tensor, at::Tensor> ClockAwareDirectLegalizeForward(
        database::PlaceDB const &                placedb,
        py::object                               pyparam,
        at::Tensor                               init_pos,
        const std::vector<std::vector<int32_t>> &inst_to_clock_indexes,
        at::Tensor                               clock_available_clock_region);

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_DIRECT_LG_SRC_DIRECT_LG_H_
