/**
 * @file   attr_object.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_ATTR_OBJECT_H_
#define OPENPARF_DATABASE_ATTR_OBJECT_H_

#include "database/object.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

template <typename T> class AttrObject : public Object {
public:
  using BaseType = Object;
  using AttrType = T;

  /// @brief default constructor
  AttrObject() : BaseType() { attr_ = new AttrType; }

  /// @brief constructor
  AttrObject(IndexType id) : BaseType(id) { attr_ = new AttrType(id); }

  /// @brief copy constructor
  AttrObject(AttrObject const &rhs) { copy(rhs); }

  /// @brief move constructor
  AttrObject(AttrObject &&rhs) { move(std::move(rhs)); }

  /// @brief copy assignment
  AttrObject &operator=(AttrObject const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  AttrObject &operator=(AttrObject &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief destructor
  ~AttrObject() { delete attr_; }

  /// @brief getter for attributes
  AttrType const &attr() const { return *attr_; }

  /// @brief getter for attributes
  AttrType &attr() { return *attr_; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const { return sizeof(attr_) + attr_->memory(); }

protected:
  /// @brief copy object
  void copy(AttrObject const &rhs) {
    this->BaseType::copy(rhs);
    *attr_ = *rhs.attr_;
  }
  /// @brief move object
  void move(AttrObject &&rhs) {
    this->BaseType::move(std::move(rhs));
    std::swap(attr_, rhs.attr_);
  }
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, AttrObject const &rhs) {
    os << className<decltype(rhs)>() << "("
       << "id_ : " << rhs.id_;
    os << ", attr_ : " << *rhs.attr_;
    os << ")";
    return os;
  }

  AttrType *attr_; ///< attributes of the object
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
