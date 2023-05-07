/**
 * File              : legality_check.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.24.2020
 * Last Modified Date: 07.24.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "ops/legality_check/src/legality_check.hpp"

#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

bool legalityCheckForward(database::PlaceDB const& placedb,
                          bool                     check_z_flag,
                          int                      max_clk_per_clock_region,
                          int                      max_clk_per_half_column,
                          at::Tensor               pos) {
  CHECK_FLAT_CPU(pos);
  CHECK_DIVISIBLE(pos, 3);
  CHECK_CONTIGUOUS(pos);

  bool legal = false;
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "LegalityCheckLauncher", [&] {
    legal = legalityCheck<scalar_t>(placedb,
                                    check_z_flag,
                                    max_clk_per_clock_region,
                                    max_clk_per_half_column,
                                    OPENPARF_TENSOR_DATA_PTR(pos, scalar_t));
  });

  return legal;
}

bool XarchLegalityCheckForward(database::PlaceDB const& placedb,
                            at::Tensor               cla_bs,
                            at::Tensor               cla_starts,
                            at::Tensor               lut_bs,
                            at::Tensor               lut_starts,
                            at::Tensor               ssr_bs,
                            at::Tensor               ssr_starts,
                            at::Tensor               pos) {
  CHECK_FLAT_CPU(pos);
  CHECK_DIVISIBLE(pos, 3);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CPU(cla_bs);
  CHECK_CONTIGUOUS(cla_bs);
  AT_ASSERT(cla_bs.dtype() == torch::kInt32);

  CHECK_FLAT_CPU(cla_starts);
  CHECK_CONTIGUOUS(cla_starts);
  AT_ASSERT(cla_starts.dtype() == torch::kInt32);

  CHECK_FLAT_CPU(lut_bs);
  CHECK_CONTIGUOUS(lut_bs);
  AT_ASSERT(lut_bs.dtype() == torch::kInt32);

  CHECK_FLAT_CPU(lut_starts);
  CHECK_CONTIGUOUS(lut_starts);
  AT_ASSERT(lut_starts.dtype() == torch::kInt32);

  CHECK_FLAT_CPU(ssr_bs);
  CHECK_CONTIGUOUS(ssr_bs);
  AT_ASSERT(ssr_bs.dtype() == torch::kInt32);

  CHECK_FLAT_CPU(ssr_starts);
  CHECK_CONTIGUOUS(ssr_starts);
  AT_ASSERT(ssr_starts.dtype() == torch::kInt32);

  bool    legal  = false;
  int32_t num_cc = cla_starts.numel() - 1;
  int32_t num_sc = ssr_starts.numel() - 1;
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "xarch legality check launcher", [&] {
    legal = xarchLegalityCheck<scalar_t>(placedb,
                                      num_cc,
                                      num_sc,
                                      OPENPARF_TENSOR_DATA_PTR(cla_bs, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(cla_starts, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(lut_bs, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(lut_starts, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(ssr_bs, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(ssr_starts, int32_t),
                                      OPENPARF_TENSOR_DATA_PTR(pos, scalar_t));
  });
  return legal;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::legalityCheckForward, "Legality check forward");
  m.def("xarchForward", &OPENPARF_NAMESPACE::XarchLegalityCheckForward, "xarch legality check forward");
}
