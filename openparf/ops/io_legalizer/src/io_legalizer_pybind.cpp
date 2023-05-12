/**
 * File              : io_legalizer_pybind.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 08.26.2021
 * Last Modified Date: 08.26.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "io_legalizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::IoLegalizerForward, "IO legalization forward");
}
