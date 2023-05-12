/**
 * @file   container.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_PYBIND_CONTAINER_H_
#define OPENPARF_PYBIND_CONTAINER_H_

// 3rd party libraries
#include "boost/optional.hpp"

// project headers
#include "container/container.hpp"
#include "pybind/util.h"

namespace container          = OPENPARF_NAMESPACE::container;

using FlatNestedVectorInt    = container::FlatNestedVector<int32_t, int32_t>;
using FlatNestedVectorUint   = container::FlatNestedVector<uint32_t, uint32_t>;
using FlatNestedVectorFloat  = container::FlatNestedVector<float, uint32_t>;
using FlatNestedVectorDouble = container::FlatNestedVector<double, uint32_t>;

// After trying many ways, I found make it work by returning the raw pointer
// and use reference_internal as the policy.
// Making the pointer like std::optional or std::shared_ptr does not work
// when binded to python. It will cause crashing.
template<typename T>
using ObserverPtr = container::ObserverPtr<T>;

// PYBIND11_DECLARE_HOLDER_TYPE(T, ObserverPtr<T>, true);

namespace pybind11 {
namespace detail {
// copied from stl.h
template<typename T>
struct type_caster<boost::optional<T>> {
  using value_conv = make_caster<typename boost::optional<T>::value_type>;

  template<typename T_>
  static handle cast(T_ &&src, return_value_policy policy, handle parent) {
    if (!src) return none().inc_ref();
    policy = return_value_policy_override<typename boost::optional<T>::value_type>::policy(policy);
    return value_conv::cast(*std::forward<T_>(src), policy, parent);
  }

  bool load(handle src, bool convert) {
    if (!src) {
      return false;
    } else if (src.is_none()) {
      return true;   // default-constructed value is already empty
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) return false;

    value.emplace(cast_op<typename boost::optional<T>::value_type &&>(std::move(inner_caster)));
    return true;
  }

  PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

// template <typename T>
// struct type_caster<ObserverPtr<T>> {
//  using value_conv = make_caster<typename ObserverPtr<T>::ValueType *>;
//
//  template <typename T_>
//  static handle cast(T_ &&src, return_value_policy policy, handle parent) {
//    if (!src) return none().inc_ref();
//    // policy = return_value_policy_override<
//    //    typename ObserverPtr<T>::ValueType>::policy(policy);
//    policy = return_value_policy::reference;
//    return value_conv::cast(*std::forward<T_>(src), policy, parent);
//  }
//
//  bool load(handle src, bool convert) {
//    if (!src) {
//      return false;
//    } else if (src.is_none()) {
//      return true;  // default-constructed value is already empty
//    }
//    value_conv inner_caster;
//    if (!inner_caster.load(src, convert)) return false;
//
//    value.emplace(
//        cast_op<typename ObserverPtr<T>::ValueType
//        *>(std::move(inner_caster)));
//    return true;
//  }
//
//  PYBIND11_TYPE_CASTER(T, _("ObserverPtr[") + value_conv::name + _("]"));
//};

// template <typename T>
// class type_caster<ObserverPtr<T>>
//    : public copyable_holder_caster<T, ObserverPtr<T>> {};

}   // namespace detail
}   // namespace pybind11

#endif
