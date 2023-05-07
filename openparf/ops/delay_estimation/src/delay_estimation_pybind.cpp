#include "delay_estimation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::delay_estimation::DelayEstimationForward, "Delay Estimation Forward");
}