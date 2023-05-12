/**
 * File              : ssr_chain_info_pybind.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "pybind/util.h"
#include "ssr_chain_info.h"

namespace py = pybind11;

using OPENPARF_NAMESPACE::ssr_chain_info::ChainInfo;
using OPENPARF_NAMESPACE::ssr_chain_info::ChainInfoVec;
using OPENPARF_NAMESPACE::ssr_chain_info::MakeChainInfoVecFromPlaceDB;
using OPENPARF_NAMESPACE::ssr_chain_info::MakeNestedNewIdx;

PYBIND11_MAKE_OPAQUE(ChainInfoVec);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ChainInfo> chain_info(m, "ChainInfo");
  chain_info.def(py::init<>());
  chain_info.def_property("inst_ids",
                          py::overload_cast<>(&ChainInfo::inst_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::inst_ids));
  chain_info.def_property("inst_new_ids",
                          py::overload_cast<>(&ChainInfo::inst_new_ids, py::const_),
                          py::overload_cast<>(&ChainInfo::inst_new_ids));
  BIND_VECTOR_TOLIST(VectorInt);
  BIND_VECTOR_TOLIST(ChainInfoVec);
  m.def("MakeChainInfoVecFromPlaceDB", &MakeChainInfoVecFromPlaceDB);
  m.def("MakeNestedNewIdx", &MakeNestedNewIdx);
}
