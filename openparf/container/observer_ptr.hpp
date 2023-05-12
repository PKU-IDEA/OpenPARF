/**
 * File              : observer_ptr.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.22.2020
 * Last Modified Date: 04.22.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_CONTAINERS_OBSERVER_PTR_HPP_
#define OPENPARF_CONTAINERS_OBSERVER_PTR_HPP_

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace container {

/// @brief A class to wrap a raw pointer such that
/// it is not exposed to python when do python binding.
/// The behavior is somewhat like std::optional<T&>,
/// which we can use to check whether a pointer is nullptr or not.
template <typename T>
class ObserverPtr {
 public:
  using ValueType = T;

  /// @brief default constructor
  ObserverPtr() : ptr_(nullptr) {}

  /// @brief constructor from pointer
  ObserverPtr(ValueType* v) : ptr_(v) {}

  /// @brief constructor from reference
  ObserverPtr(ValueType& v) : ptr_(&v) {}

  /// @brief copy constructor
  ObserverPtr(ObserverPtr const& rhs) { ptr_ = rhs.ptr_; }

  /// @brief move constructor
  // ObserverPtr(ObserverPtr&& rhs) { ptr_ = rhs.ptr_; }

  /// @brief copy assignment
  ObserverPtr& operator=(ObserverPtr const& rhs) {
    if (this != &rhs) {
      ptr_ = rhs.ptr_;
    }
    return *this;
  }

  /// @brief move assignment
  // ObserverPtr& operator=(ObserverPtr&& rhs) {
  //  if (this != &rhs) {
  //    ptr_ = std::move(rhs.ptr_);
  //  }
  //  return *this;
  //}

  /// @brief getter for the pointer
  ValueType* get() const { return ptr_; }

  /// @brief getter for the reference
  ValueType& value() const { return *ptr_; }

  /// @brief overload dereference
  ValueType& operator*() const { return *ptr_; }

  /// @brief overload dereference
  ValueType* operator->() const { return ptr_; }

  /// @brief operator bool
  explicit operator bool() const { return (ptr_ != nullptr); }

  /// @brief overload operator equality
  bool operator==(ObserverPtr const& rhs) const { return ptr_ == rhs.ptr_; }

  /// @brief overload operator less
  bool operator<(ObserverPtr const& rhs) const { return ptr_ < rhs.ptr_; }

  /// @brief overload operator greater
  bool operator>(ObserverPtr const& rhs) const { return ptr_ > rhs.ptr_; }

  /// @brief overload operator less equal
  bool operator<=(ObserverPtr const& rhs) const { return ptr_ <= rhs.ptr_; }

  /// @brief overload operator greater equal
  bool operator>=(ObserverPtr const& rhs) const { return ptr_ >= rhs.ptr_; }

 protected:
  /// @brief overload operator equality
  friend bool operator==(ObserverPtr const& a, ValueType const* b) {
    return a.ptr_ == b;
  }
  /// @brief overload operator equality
  friend bool operator==(ValueType const* a, ObserverPtr const& b) {
    return b.ptr_ == a;
  }
  /// @brief overload operator equality
  friend bool operator==(std::nullptr_t a, ObserverPtr const& b) {
    return b.ptr_ == a;
  }
  /// @brief overload operator equality
  friend bool operator==(ObserverPtr const& a, std::nullptr_t b) {
    return a.ptr_ == b;
  }

  /// @brief overload operator less
  friend bool operator<(ObserverPtr const& a, ValueType const* b) {
    return a.ptr_ < b;
  }
  /// @brief overload operator less
  friend bool operator<(ValueType const* a, ObserverPtr const& b) {
    return a < b.ptr_;
  }
  /// @brief overload operator less
  friend bool operator<(ObserverPtr const& a, std::nullptr_t b) {
    return a.ptr_ < b;
  }
  /// @brief overload operator less
  friend bool operator<(std::nullptr_t a, ObserverPtr const& b) {
    return a < b.ptr_;
  }

  /// @brief overload operator greater
  friend bool operator>(ObserverPtr const& a, ValueType const* b) {
    return a.ptr_ > b;
  }
  /// @brief overload operator greater
  friend bool operator>(ValueType const* a, ObserverPtr const& b) {
    return a > b.ptr_;
  }
  /// @brief overload operator greater
  friend bool operator>(ObserverPtr const& a, std::nullptr_t b) {
    return a.ptr_ > b;
  }
  /// @brief overload operator greater
  friend bool operator>(std::nullptr_t a, ObserverPtr const& b) {
    return a > b.ptr_;
  }

  /// @brief overload operator less equal
  friend bool operator<=(ObserverPtr const& a, ValueType const* b) {
    return a.ptr_ <= b;
  }
  /// @brief overload operator less equal
  friend bool operator<=(ValueType const* a, ObserverPtr const& b) {
    return a <= b.ptr_;
  }
  /// @brief overload operator less equal
  friend bool operator<=(ObserverPtr const& a, std::nullptr_t b) {
    return a.ptr_ <= b;
  }
  /// @brief overload operator less equal
  friend bool operator<=(std::nullptr_t a, ObserverPtr const& b) {
    return a <= b.ptr_;
  }

  /// @brief overload operator greater equal
  friend bool operator>=(ObserverPtr const& a, ValueType const* b) {
    return a.ptr_ >= b;
  }
  /// @brief overload operator greater equal
  friend bool operator>=(ValueType const* a, ObserverPtr const& b) {
    return a >= b.ptr_;
  }
  /// @brief overload operator greater equal
  friend bool operator>=(ObserverPtr const& a, std::nullptr_t b) {
    return a.ptr_ >= b;
  }
  /// @brief overload operator greater equal
  friend bool operator>=(std::nullptr_t a, ObserverPtr const& b) {
    return a >= b.ptr_;
  }

  ValueType* ptr_;  ///< raw pointer
};

}  // namespace container

OPENPARF_END_NAMESPACE

#endif
