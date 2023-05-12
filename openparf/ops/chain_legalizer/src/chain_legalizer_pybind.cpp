/**
 * File              : chain_legalizer_pybind.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.10.2021
 * Last Modified Date: 09.10.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "chain_legalizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::chain_legalizer::ChainLegalizerForward, "Chain legalization forward");
}
