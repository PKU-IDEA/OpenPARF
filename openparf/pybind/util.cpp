/**
 * @file   util.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "pybind/util.h"
#include "pybind/geometry.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(VectorModelType);
PYBIND11_MAKE_OPAQUE(VectorSignalDirection);
PYBIND11_MAKE_OPAQUE(VectorSignalType);
PYBIND11_MAKE_OPAQUE(VectorShapeConstrType);
PYBIND11_MAKE_OPAQUE(VectorPlaceStatus);
PYBIND11_MAKE_OPAQUE(VectorResourceCategory);

PYBIND11_MAKE_OPAQUE(VectorChar);
PYBIND11_MAKE_OPAQUE(VectorUchar);
PYBIND11_MAKE_OPAQUE(VectorInt);
PYBIND11_MAKE_OPAQUE(VectorUint);
PYBIND11_MAKE_OPAQUE(VectorFloat);
PYBIND11_MAKE_OPAQUE(VectorDouble);
PYBIND11_MAKE_OPAQUE(VectorString);

PYBIND11_MAKE_OPAQUE(VectorCharL2);
PYBIND11_MAKE_OPAQUE(VectorUcharL2);
PYBIND11_MAKE_OPAQUE(VectorIntL2);
PYBIND11_MAKE_OPAQUE(VectorUintL2);
PYBIND11_MAKE_OPAQUE(VectorFloatL2);
PYBIND11_MAKE_OPAQUE(VectorDoubleL2);

void bind_util(py::module& m) {
  m.doc() = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: util
        .. autosummary::
           :toctree: _generate
           MessageType
           ModelType
           SignalDirection
           SignalType
           ShapeConstrType
           PlaceStatus
           ResourceCategory
    )pbdoc";

  py::enum_<MessageType>(m, "MessageType")
      .value("kNone", MessageType::kNone)
      .value("kInfo", MessageType::kInfo)
      .value("kWarn", MessageType::kWarn)
      .value("kError", MessageType::kError)
      .value("kDebug", MessageType::kDebug)
      .value("kAssert", MessageType::kAssert);

  py::enum_<ModelType>(m, "ModelType")
      .value("kCell", ModelType::kCell)
      .value("kIOPin", ModelType::kIOPin)
      .value("kModule", ModelType::kModule)
      .value("kUnknown", ModelType::kUnknown);

  py::enum_<SignalDirection>(m, "SignalDirection")
      .value("kInput", SignalDirection::kInput)
      .value("kOutput", SignalDirection::kOutput)
      .value("kInout", SignalDirection::kInout)
      .value("kUnknown", SignalDirection::kUnknown);

  py::enum_<SignalType>(m, "SignalType")
      .value("kClock", SignalType::kClock)
      .value("kControlSR", SignalType::kControlSR)
      .value("kControlCE", SignalType::kControlCE)
      .value("kCascade", SignalType::kCascade)
      .value("kSignal", SignalType::kSignal)
      .value("kUnknown", SignalType::kUnknown);

  py::enum_<ShapeConstrType>(m, "ShapeConstrType")
      .value("kDSPChain", ShapeConstrType::kDSPChain)
      .value("kRAMChain", ShapeConstrType::kRAMChain)
      .value("kCarryChain", ShapeConstrType::kCarryChain)
      .value("kUnknown", ShapeConstrType::kUnknown);

  py::enum_<PlaceStatus>(m, "PlaceStatus")
      .value("kMovable", PlaceStatus::kMovable)
      .value("kFixed", PlaceStatus::kFixed)
      .value("kUnknown", PlaceStatus::kUnknown);

  py::enum_<ResourceCategory>(m, "ResourceCategory")
      .value("kLUTL", ResourceCategory::kLUTL)
      .value("kLUTM", ResourceCategory::kLUTM)
      .value("kFF", ResourceCategory::kFF)
      .value("kCarry", ResourceCategory::kCarry)
      .value("kSSSIR", ResourceCategory::kSSSIR)
      .value("kSSMIR", ResourceCategory::kSSMIR)
      .value("kUnknown", ResourceCategory::kUnknown)
      ;

  //py::class_<ExtensibleEnum>(m, "ExtensibleEnum")
  //    .def(py::init<>())
  //    .def(py::init<ExtensibleEnum::NameMapType const&,
  //                  ExtensibleEnum::AliasMapType const&>())
  //    .def("size", &ExtensibleEnum::size)
  //    .def("sizeInitial", &ExtensibleEnum::sizeInitial)
  //    .def("sizeExt", &ExtensibleEnum::sizeExt)
  //    .def("id",
  //         (ExtensibleEnum::IndexType(ExtensibleEnum::*)(std::string const&)) &
  //             ExtensibleEnum::id)
  //    .def("name",
  //         (std::string const& (ExtensibleEnum::*)(ExtensibleEnum::IndexType)) &
  //             ExtensibleEnum::name)
  //    .def("__iter__",
  //         [](ExtensibleEnum const& rhs) {
  //           return py::make_iterator(rhs.begin(), rhs.end());
  //         })
  //    .def("__repr__", [](ExtensibleEnum const& rhs) {
  //      std::stringstream os;
  //      os << rhs;
  //      return os.str();
  //    });

  //py::class_<ResourceTypeEnum, ExtensibleEnum>(m, "ResourceTypeEnum")
  //    .def(py::init<>())
  //    .def("__repr__", [](ResourceTypeEnum const& rhs) {
  //      std::stringstream os;
  //      os << rhs;
  //      return os.str();
  //    });

  //py::class_<AreaTypeEnum, ExtensibleEnum>(m, "AreaTypeEnum")
  //    .def(py::init<>())
  //    .def("initialize", (void (AreaTypeEnum::*)(ResourceTypeEnum const&)) &
  //                           AreaTypeEnum::initialize)
  //    .def("resourceAreaTypes",
  //         (std::vector<std::string>(AreaTypeEnum::*)(std::string const&)) &
  //             AreaTypeEnum::resourceAreaTypes)
  //    .def("resourceAreaTypeIds", (std::vector<AreaTypeEnum::IndexType>(
  //                                    AreaTypeEnum::*)(std::string const&)) &
  //                                    AreaTypeEnum::resourceAreaTypeIds)
  //    .def("__repr__", [](AreaTypeEnum const& rhs) {
  //      std::stringstream os;
  //      os << rhs;
  //      return os.str();
  //    });

  //py::class_<Arch>(m, "Arch")
  //  .def(py::init<std::vector<std::vector<uint8_t>> const&, std::vector<uint8_t> const&, std::vector<int32_t> const&>())
  //  .def("numModels", &Arch::numModels)
  //  .def("numResources", &Arch::numResources)
  //  .def("numCategories", &Arch::numCategories)
  //  .def("numSites", &Arch::numSites)
  //  .def("siteResourceCapacity", (int32_t (Arch::*)(uint32_t, uint8_t)) &Arch::siteResourceCapacity)
  //  .def("model2Resources", (std::vector<uint8_t> const& (Arch::*)(uint32_t)) &Arch::model2Resources)
  //  .def("resource2Models", (std::vector<uint32_t> const& (Arch::*)(uint8_t)) &Arch::resource2Models)
  //  .def("resource2Category", (uint8_t (Arch::*)(uint8_t)) &Arch::resource2Category)
  //  .def("category2Resources", (std::vector<uint8_t> const& (Arch::*)(uint8_t)) &Arch::category2Resources)
  //  .def("model2Categories", (std::vector<uint8_t> const& (Arch::*)(uint32_t)) &Arch::model2Categories)
  //  .def("siteCategoryCapacity", [](Arch const& rhs, uint32_t site, uint8_t category){
  //        py::list pycaps;
  //        auto caps = rhs.siteCategoryCapacity(site, category);
  //        for (auto const& cap : caps) {
  //          pycaps.append(py::make_tuple(cap.first, cap.second));
  //        }
  //        return pycaps;
  //      })
  //  ;

  /// @brief bind print function
  m.def("ediPrint",
        [](MessageType m, std::string const& msg) {
          OPENPARF_NAMESPACE::openparfPrint(m, "%s", msg.c_str());
        },
        "Print message");

  BIND_VECTOR_TOLIST(VectorModelType);
  BIND_VECTOR_TOLIST(VectorSignalDirection);
  BIND_VECTOR_TOLIST(VectorSignalType);
  BIND_VECTOR_TOLIST(VectorShapeConstrType);
  BIND_VECTOR_TOLIST(VectorPlaceStatus);
  BIND_VECTOR_TOLIST(VectorResourceCategory);

  BIND_VECTOR_TOLIST(VectorChar);
  BIND_VECTOR_TOLIST(VectorUchar);
  BIND_VECTOR_TOLIST(VectorInt);
  BIND_VECTOR_TOLIST(VectorUint);
  BIND_VECTOR_TOLIST(VectorLong);
  BIND_VECTOR_TOLIST(VectorUlong);
  BIND_VECTOR_TOLIST(VectorFloat);
  BIND_VECTOR_TOLIST(VectorDouble);
  BIND_VECTOR_TOLIST(VectorString);

  BIND_VECTOR_TOLIST_LEVELS(VectorCharL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorUcharL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorIntL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorUintL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorLongL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorUlongL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorFloatL2, 2);
  BIND_VECTOR_TOLIST_LEVELS(VectorDoubleL2, 2);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "develop";
#endif
}
