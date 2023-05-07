/**
 * @file   object.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_OBJECT_H_
#define OPENPARF_DATABASE_OBJECT_H_

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

class Object {
public:
  using CoordinateType = CoordinateTraits<int32_t>::CoordinateType;
  using IndexType = CoordinateTraits<CoordinateType>::IndexType;

  /// @brief default constructor
  Object() { setId(std::numeric_limits<IndexType>::max()); }

  /// @brief constructor
  Object(IndexType id) : id_(id) {}

  /// @brief copy constructor
  Object(Object const &rhs) { copy(rhs); }

  /// @brief move constructor
  Object(Object &&rhs) { move(std::move(rhs)); }

  /// @brief copy assignment
  Object &operator=(Object const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Object &operator=(Object &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief destructor
  virtual ~Object() {}

  /// @brief get id of the object
  /// @return id
  IndexType id() const { return id_; }

  /// @brief set id of the object
  /// @param id
  void setId(IndexType id) { id_ = id; }

  /// @brief summarize memory usage of the object in bytes
  virtual IndexType memory() const { return sizeof(*this); }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Object const &rhs) {
    os << className<decltype(rhs)>() << "("
       << "id_ : " << rhs.id_ << ")";
    return os;
  }

protected:
  /// @brief copy object
  void copy(Object const &rhs);
  /// @brief move object
  void move(Object &&rhs);

  IndexType id_;
};

template <typename T>
struct OutputChar {
  T const& operator()(T const& v) const {
    return v;
  }
};

template <>
struct OutputChar<int8_t> {
  int32_t operator()(int8_t const& v) const {
    return v;
  }
};

template <>
struct OutputChar<uint8_t> {
  uint32_t operator()(uint8_t const& v) const {
    return v;
  }
};

/// @brief output stream for vector
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &rhs) {
  os << "[" << rhs.size() << "][";
  const char *delimiter = "";
  for (auto v : rhs) {
    os << delimiter << OutputChar<T>()(v);
    delimiter = ", ";
  }
  os << "]";
  return os;
}

/// @brief output stream for vector
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T *> const &rhs) {
  os << "[" << rhs.size() << "][";
  const char *delimiter = "";
  for (auto v : rhs) {
    os << delimiter << OutputChar<T>()(*v);
    delimiter = ", ";
  }
  os << "]";
  return os;
}

/// @brief input stream for vector
/// Assume the format is size + content
template <typename T>
std::istream &operator>>(std::istream &is, std::vector<T> &rhs) {
  Object::IndexType size;
  is >> size;
  rhs.resize(size);
  for (Object::IndexType i = 0; i < size; ++i) {
    is >> rhs[i];
  }
  return is;
}

} // namespace database

OPENPARF_END_NAMESPACE

#endif
