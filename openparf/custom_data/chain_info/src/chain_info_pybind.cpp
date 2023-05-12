#include "custom_data/chain_info/src/chain_info.h"
#include "pybind/util.h"

namespace py = pybind11;

using OPENPARF_NAMESPACE::chain_info::ChainInfo;
using OPENPARF_NAMESPACE::chain_info::ChainInfoVec;
using OPENPARF_NAMESPACE::chain_info::MakeChainInfoVecFromPlaceDB;
using OPENPARF_NAMESPACE::chain_info::MakeNestedNewIdx;

PYBIND11_MAKE_OPAQUE(ChainInfoVec);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ChainInfo> chain_info(m, "ChainInfo");
  chain_info.def(py::init<>());
  chain_info.def_property("cla_ids",
                          py::overload_cast<>(&ChainInfo::cla_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::cla_ids));
  chain_info.def_property("lut_ids",
                          py::overload_cast<>(&ChainInfo::lut_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::lut_ids));
  chain_info.def_property("cla_new_ids",
                          py::overload_cast<>(&ChainInfo::cla_new_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::cla_new_ids));
  chain_info.def_property("lut_new_ids",
                          py::overload_cast<>(&ChainInfo::lut_new_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::lut_new_ids));
  BIND_VECTOR_TOLIST(VectorInt);
  BIND_VECTOR_TOLIST(ChainInfoVec);
  m.def("MakeChainInfoVecFromPlaceDB", &MakeChainInfoVecFromPlaceDB);
  m.def("MakeNestedNewIdx", &MakeNestedNewIdx);
}
