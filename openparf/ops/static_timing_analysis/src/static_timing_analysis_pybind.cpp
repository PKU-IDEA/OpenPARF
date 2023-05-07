#include "static_timing_analysis.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",
        &OPENPARF_NAMESPACE::static_timing_analysis::StaticTimingAnalysisForward,
        "Static Timing Analysis Forward");
}