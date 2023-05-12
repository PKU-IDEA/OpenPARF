/**
 * @file   container.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "pybind/container.h"

#include <sstream>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(FlatNestedVectorInt);
PYBIND11_MAKE_OPAQUE(FlatNestedVectorUint);
PYBIND11_MAKE_OPAQUE(FlatNestedVectorFloat);
PYBIND11_MAKE_OPAQUE(FlatNestedVectorDouble);

void bind_container(py::module &m) {
  m.doc() = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: container
        .. autosummary::
           :toctree: _generate
           FlatNestedVector
           MultiDimMap
    )pbdoc";

  py::class_<FlatNestedVectorInt> flat_nested_vector_int(m,
                                                         "FlatNestedVectorInt");
  flat_nested_vector_int.def(py::init<>())
      .def(py::init<
           std::vector<std::vector<FlatNestedVectorInt::ValueType>> const &>())
      .def("size", &FlatNestedVectorInt::size)
      .def("size1", &FlatNestedVectorInt::size1)
      .def("size2", (FlatNestedVectorInt::IndexType(FlatNestedVectorInt::*)(
                        FlatNestedVectorInt::IndexType)) &
                        FlatNestedVectorInt::size2)
      .def("capacity", &FlatNestedVectorInt::capacity)
      .def("capacity1", &FlatNestedVectorInt::capacity1)
      .def("reserve",
           (void (FlatNestedVectorInt::*)(FlatNestedVectorInt::IndexType,
                                          FlatNestedVectorInt::IndexType)) &
               FlatNestedVectorInt::reserve)
      .def("at",
           py::overload_cast<FlatNestedVectorInt::IndexType>(
               &FlatNestedVectorInt::at),
           "Return the iterator pair of the inner vector range")
      .def("at",
           py::overload_cast<FlatNestedVectorInt::IndexType,
                             FlatNestedVectorInt::IndexType>(
               &FlatNestedVectorInt::at),
           py::return_value_policy::reference_internal, "Return one element")
      .def("data",
           (std::vector<FlatNestedVectorInt::ValueType> &
            (FlatNestedVectorInt::*)()) &
               FlatNestedVectorInt::data,
           py::return_value_policy::reference_internal)
      .def("indexBeginData",
           (std::vector<FlatNestedVectorInt::IndexType> &
            (FlatNestedVectorInt::*)()) &
               FlatNestedVectorInt::indexBeginData,
           py::return_value_policy::reference_internal)
      .def("indexBeginAt",
           (FlatNestedVectorInt::IndexType(FlatNestedVectorInt::*)(
               FlatNestedVectorInt::IndexType)) &
               FlatNestedVectorInt::indexBeginAt)
      .def("indexEndAt",
           (FlatNestedVectorInt::IndexType(FlatNestedVectorInt::*)(
               FlatNestedVectorInt::IndexType)) &
               FlatNestedVectorInt::indexEndAt)
      .def("pushBackIndexBegin",
           (void (FlatNestedVectorInt::*)(FlatNestedVectorInt::IndexType)) &
               FlatNestedVectorInt::pushBackIndexBegin)
      .def("pushBack", (void (FlatNestedVectorInt::*)(
                           FlatNestedVectorInt::ValueType const &)) &
                           FlatNestedVectorInt::pushBack)
      .def("clear", &FlatNestedVectorInt::clear)
      .def("__iter__",
           [](FlatNestedVectorInt &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>())
      .def("__repr__", [](FlatNestedVectorInt const &rhs) {
        std::stringstream os;
        os << rhs;
        return os.str();
      });

  py::class_<FlatNestedVectorInt::VectorProxy>(flat_nested_vector_int,
                                               "VectorProxy")
      .def(py::init<FlatNestedVectorInt::IteratorType,
                    FlatNestedVectorInt::IteratorType>())
      .def("size", (FlatNestedVectorInt::IndexType(
                       FlatNestedVectorInt::VectorProxy::*)()) &
                       FlatNestedVectorInt::VectorProxy::size)
      .def("__iter__",
           [](FlatNestedVectorInt::VectorProxy &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>());

  py::class_<FlatNestedVectorUint> flat_nested_vector_uint(
      m, "FlatNestedVectorUint");
  flat_nested_vector_uint.def(py::init<>())
      .def(py::init<
           std::vector<std::vector<FlatNestedVectorUint::ValueType>> const &>())
      .def("size", &FlatNestedVectorUint::size)
      .def("size1", &FlatNestedVectorUint::size1)
      .def("size2", (FlatNestedVectorUint::IndexType(FlatNestedVectorUint::*)(
                        FlatNestedVectorUint::IndexType)) &
                        FlatNestedVectorUint::size2)
      .def("capacity", &FlatNestedVectorUint::capacity)
      .def("capacity1", &FlatNestedVectorUint::capacity1)
      .def("reserve",
           (void (FlatNestedVectorUint::*)(FlatNestedVectorUint::IndexType,
                                           FlatNestedVectorUint::IndexType)) &
               FlatNestedVectorUint::reserve)
      .def("at",
           py::overload_cast<FlatNestedVectorUint::IndexType>(
               &FlatNestedVectorUint::at),
           "Return the iterator pair of the inner vector range")
      .def("at",
           py::overload_cast<FlatNestedVectorUint::IndexType,
                             FlatNestedVectorUint::IndexType>(
               &FlatNestedVectorUint::at),
           py::return_value_policy::reference_internal, "Return one element")
      .def("data",
           (std::vector<FlatNestedVectorUint::ValueType> &
            (FlatNestedVectorUint::*)()) &
               FlatNestedVectorUint::data,
           py::return_value_policy::reference_internal)
      .def("indexBeginData",
           (std::vector<FlatNestedVectorUint::IndexType> &
            (FlatNestedVectorUint::*)()) &
               FlatNestedVectorUint::indexBeginData,
           py::return_value_policy::reference_internal)
      .def("indexBeginAt",
           (FlatNestedVectorUint::IndexType(FlatNestedVectorUint::*)(
               FlatNestedVectorUint::IndexType)) &
               FlatNestedVectorUint::indexBeginAt)
      .def("indexEndAt",
           (FlatNestedVectorUint::IndexType(FlatNestedVectorUint::*)(
               FlatNestedVectorUint::IndexType)) &
               FlatNestedVectorUint::indexEndAt)
      .def("pushBackIndexBegin",
           (void (FlatNestedVectorUint::*)(FlatNestedVectorUint::IndexType)) &
               FlatNestedVectorUint::pushBackIndexBegin)
      .def("pushBack", (void (FlatNestedVectorUint::*)(
                           FlatNestedVectorUint::ValueType const &)) &
                           FlatNestedVectorUint::pushBack)
      .def("clear", &FlatNestedVectorUint::clear)
      .def("__iter__",
           [](FlatNestedVectorUint &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>())
      .def("__repr__", [](FlatNestedVectorUint const &rhs) {
        std::stringstream os;
        os << rhs;
        return os.str();
      });

  py::class_<FlatNestedVectorUint::VectorProxy>(flat_nested_vector_uint,
                                                "VectorProxy")
      .def(py::init<FlatNestedVectorUint::IteratorType,
                    FlatNestedVectorUint::IteratorType>())
      .def("size", (FlatNestedVectorUint::IndexType(
                       FlatNestedVectorUint::VectorProxy::*)()) &
                       FlatNestedVectorUint::VectorProxy::size)
      .def("__iter__",
           [](FlatNestedVectorUint::VectorProxy &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>());

  py::class_<FlatNestedVectorFloat> flat_nested_vector_float(
      m, "FlatNestedVectorFloat");
  flat_nested_vector_float.def(py::init<>())
      .def(py::init<std::vector<
               std::vector<FlatNestedVectorFloat::ValueType>> const &>())
      .def("size", &FlatNestedVectorFloat::size)
      .def("size1", &FlatNestedVectorFloat::size1)
      .def("size2", (FlatNestedVectorFloat::IndexType(FlatNestedVectorFloat::*)(
                        FlatNestedVectorFloat::IndexType)) &
                        FlatNestedVectorFloat::size2)
      .def("capacity", &FlatNestedVectorFloat::capacity)
      .def("capacity1", &FlatNestedVectorFloat::capacity1)
      .def("reserve",
           (void (FlatNestedVectorFloat::*)(FlatNestedVectorFloat::IndexType,
                                            FlatNestedVectorFloat::IndexType)) &
               FlatNestedVectorFloat::reserve)
      .def("at",
           py::overload_cast<FlatNestedVectorFloat::IndexType>(
               &FlatNestedVectorFloat::at),
           "Return the iterator pair of the inner vector range")
      .def("at",
           py::overload_cast<FlatNestedVectorFloat::IndexType,
                             FlatNestedVectorFloat::IndexType>(
               &FlatNestedVectorFloat::at),
           py::return_value_policy::reference_internal, "Return one element")
      .def("data",
           (std::vector<FlatNestedVectorFloat::ValueType> &
            (FlatNestedVectorFloat::*)()) &
               FlatNestedVectorFloat::data,
           py::return_value_policy::reference_internal)
      .def("indexBeginData",
           (std::vector<FlatNestedVectorFloat::IndexType> &
            (FlatNestedVectorFloat::*)()) &
               FlatNestedVectorFloat::indexBeginData,
           py::return_value_policy::reference_internal)
      .def("indexBeginAt",
           (FlatNestedVectorFloat::IndexType(FlatNestedVectorFloat::*)(
               FlatNestedVectorFloat::IndexType)) &
               FlatNestedVectorFloat::indexBeginAt)
      .def("indexEndAt",
           (FlatNestedVectorFloat::IndexType(FlatNestedVectorFloat::*)(
               FlatNestedVectorFloat::IndexType)) &
               FlatNestedVectorFloat::indexEndAt)
      .def("pushBackIndexBegin",
           (void (FlatNestedVectorFloat::*)(FlatNestedVectorFloat::IndexType)) &
               FlatNestedVectorFloat::pushBackIndexBegin)
      .def("pushBack", (void (FlatNestedVectorFloat::*)(
                           FlatNestedVectorFloat::ValueType const &)) &
                           FlatNestedVectorFloat::pushBack)
      .def("clear", &FlatNestedVectorFloat::clear)
      .def("__iter__",
           [](FlatNestedVectorFloat &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>())
      .def("__repr__", [](FlatNestedVectorFloat const &rhs) {
        std::stringstream os;
        os << rhs;
        return os.str();
      });

  py::class_<FlatNestedVectorFloat::VectorProxy>(flat_nested_vector_float,
                                                 "VectorProxy")
      .def(py::init<FlatNestedVectorFloat::IteratorType,
                    FlatNestedVectorFloat::IteratorType>())
      .def("size", (FlatNestedVectorFloat::IndexType(
                       FlatNestedVectorFloat::VectorProxy::*)()) &
                       FlatNestedVectorFloat::VectorProxy::size)
      .def("__iter__",
           [](FlatNestedVectorFloat::VectorProxy &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>());

  py::class_<FlatNestedVectorDouble> flat_nested_vector_double(
      m, "FlatNestedVectorDouble");
  flat_nested_vector_double.def(py::init<>())
      .def(py::init<std::vector<
               std::vector<FlatNestedVectorDouble::ValueType>> const &>())
      .def("size", &FlatNestedVectorDouble::size)
      .def("size1", &FlatNestedVectorDouble::size1)
      .def("size2",
           (FlatNestedVectorDouble::IndexType(FlatNestedVectorDouble::*)(
               FlatNestedVectorDouble::IndexType)) &
               FlatNestedVectorDouble::size2)
      .def("capacity", &FlatNestedVectorDouble::capacity)
      .def("capacity1", &FlatNestedVectorDouble::capacity1)
      .def("reserve", (void (FlatNestedVectorDouble::*)(
                          FlatNestedVectorDouble::IndexType,
                          FlatNestedVectorDouble::IndexType)) &
                          FlatNestedVectorDouble::reserve)
      .def("at",
           py::overload_cast<FlatNestedVectorDouble::IndexType>(
               &FlatNestedVectorDouble::at),
           "Return the iterator pair of the inner vector range")
      .def("at",
           py::overload_cast<FlatNestedVectorDouble::IndexType,
                             FlatNestedVectorDouble::IndexType>(
               &FlatNestedVectorDouble::at),
           py::return_value_policy::reference_internal, "Return one element")
      .def("data",
           (std::vector<FlatNestedVectorDouble::ValueType> &
            (FlatNestedVectorDouble::*)()) &
               FlatNestedVectorDouble::data,
           py::return_value_policy::reference_internal)
      .def("indexBeginData",
           (std::vector<FlatNestedVectorDouble::IndexType> &
            (FlatNestedVectorDouble::*)()) &
               FlatNestedVectorDouble::indexBeginData,
           py::return_value_policy::reference_internal)
      .def("indexBeginAt",
           (FlatNestedVectorDouble::IndexType(FlatNestedVectorDouble::*)(
               FlatNestedVectorDouble::IndexType)) &
               FlatNestedVectorDouble::indexBeginAt)
      .def("indexEndAt",
           (FlatNestedVectorDouble::IndexType(FlatNestedVectorDouble::*)(
               FlatNestedVectorDouble::IndexType)) &
               FlatNestedVectorDouble::indexEndAt)
      .def("pushBackIndexBegin", (void (FlatNestedVectorDouble::*)(
                                     FlatNestedVectorDouble::IndexType)) &
                                     FlatNestedVectorDouble::pushBackIndexBegin)
      .def("pushBack", (void (FlatNestedVectorDouble::*)(
                           FlatNestedVectorDouble::ValueType const &)) &
                           FlatNestedVectorDouble::pushBack)
      .def("clear", &FlatNestedVectorDouble::clear)
      .def("__iter__",
           [](FlatNestedVectorDouble &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>())
      .def("__repr__", [](FlatNestedVectorDouble const &rhs) {
        std::stringstream os;
        os << rhs;
        return os.str();
      });

  py::class_<FlatNestedVectorDouble::VectorProxy>(flat_nested_vector_double,
                                                  "VectorProxy")
      .def(py::init<FlatNestedVectorDouble::IteratorType,
                    FlatNestedVectorDouble::IteratorType>())
      .def("size", (FlatNestedVectorDouble::IndexType(
                       FlatNestedVectorDouble::VectorProxy::*)()) &
                       FlatNestedVectorDouble::VectorProxy::size)
      .def("__iter__",
           [](FlatNestedVectorDouble::VectorProxy &rhs) {
             return py::make_iterator(rhs.begin(), rhs.end());
           },
           py::keep_alive<0, 1>());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "develop";
#endif
}
