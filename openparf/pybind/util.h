/**
 * @file   util.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_PYBIND_UTIL_H_
#define OPENPARF_PYBIND_UTIL_H_

#include <string>
#include <vector>

#include "pybind11/pybind11.h"
// It looks like stl.h and stl_bind.h cannot be included at the same time.
// I need to bind vectors, but use type conversion for others.
// As a result, I choose to do the type conversion manually.
#include "pybind11/stl_bind.h"
#include "util/util.h"

using MessageType            = OPENPARF_NAMESPACE::MessageType;
using ModelType              = OPENPARF_NAMESPACE::ModelType;
using VectorModelType        = std::vector<ModelType>;
using SignalDirection        = OPENPARF_NAMESPACE::SignalDirection;
using VectorSignalDirection  = std::vector<SignalDirection>;
using SignalType             = OPENPARF_NAMESPACE::SignalType;
using VectorSignalType       = std::vector<SignalType>;
using ShapeConstrType        = OPENPARF_NAMESPACE::ShapeConstrType;
using VectorShapeConstrType  = std::vector<ShapeConstrType>;
using PlaceStatus            = OPENPARF_NAMESPACE::PlaceStatus;
using VectorPlaceStatus      = std::vector<PlaceStatus>;
using ResourceCategory       = OPENPARF_NAMESPACE::ResourceCategory;
using VectorResourceCategory = std::vector<ResourceCategory>;

// using ExtensibleEnum = OPENPARF_NAMESPACE::ExtensibleEnum;
// using ResourceTypeEnum = OPENPARF_NAMESPACE::ResourceTypeEnum;
// using AreaTypeEnum = OPENPARF_NAMESPACE::AreaTypeEnum;

// using Arch = OPENPARF_NAMESPACE::Arch;

using VectorChar             = std::vector<int8_t>;
using VectorUchar            = std::vector<uint8_t>;
using VectorInt              = std::vector<int32_t>;
using VectorUint             = std::vector<uint32_t>;
using VectorLong             = std::vector<int64_t>;
using VectorUlong            = std::vector<uint64_t>;
using VectorFloat            = std::vector<float>;
using VectorDouble           = std::vector<double>;
using VectorString           = std::vector<std::string>;

using VectorCharL2           = std::vector<std::vector<int8_t>>;
using VectorUcharL2          = std::vector<std::vector<uint8_t>>;
using VectorIntL2            = std::vector<std::vector<int32_t>>;
using VectorUintL2           = std::vector<std::vector<uint32_t>>;
using VectorLongL2           = std::vector<std::vector<int64_t>>;
using VectorUlongL2          = std::vector<std::vector<uint64_t>>;
using VectorFloatL2          = std::vector<std::vector<float>>;
using VectorDoubleL2         = std::vector<std::vector<double>>;

template<typename T, std::size_t Level>
struct NestedVector2List {
  using type = T;
  pybind11::list operator()(type const& rhs) const {
    pybind11::list                                          ret(rhs.size());
    NestedVector2List<typename type::value_type, Level - 1> conv;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      ret[i] = conv(rhs[i]);
    }
    return ret;
  }
};
template<typename T>
struct NestedVector2List<T, 0> {
  using type = T;
  type const& operator()(type const& rhs) const { return rhs; }
};

#define BIND_VECTOR_TOLIST(T)                                                                                          \
  py::bind_vector<T>(m, #T)                                                                                            \
          .def("tolist",                                                                                               \
               [](T const& rhs) {                                                                                      \
                 py::list ret(rhs.size());                                                                             \
                 for (std::size_t i = 0; i < rhs.size(); ++i) {                                                        \
                   ret[i] = rhs[i];                                                                                    \
                 }                                                                                                     \
                 return ret;                                                                                           \
               })                                                                                                      \
          .def("size", [](T& rhs) { return rhs.size(); })                                                              \
          .def("resize", [](T& rhs, std::size_t n) { rhs.resize(n); })                                                 \
          .def("assign", [](T& rhs, std::size_t n, T::value_type const& v) { rhs.assign(n, v); })

#define BIND_VECTOR_TOLIST_LEVELS(T, Level)                                                                            \
  py::bind_vector<T>(m, #T)                                                                                            \
          .def("tolist",                                                                                               \
               [](T const& rhs) {                                                                                      \
                 NestedVector2List<T, Level> conv;                                                                     \
                 return conv(rhs);                                                                                     \
               })                                                                                                      \
          .def("resize", [](T& rhs, std::size_t n) { rhs.resize(n); })

#endif   // OPENPARF_PYBIND_UTIL_H_
