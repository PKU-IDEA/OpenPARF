/**
 * File              : ssr_abacus_legalizer_pybind.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ssr_abacus_legalizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::ssr_abacus_legalizer::SsrLegalizerForward, "SSR Legalization forward");
}
