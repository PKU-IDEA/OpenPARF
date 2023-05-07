/**
 * @file   layout_map.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_LAYOUT_MAP_HPP_
#define OPENPARF_DATABASE_LAYOUT_MAP_HPP_

#include "container/container.hpp"
#include "database/object.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

template <typename T> class LayoutMap2D : public Object {
public:
  using ValueType = T;
  using BaseType = Object;
  using MapType = container::MultiDimMap<ValueType, 2, false>;
  using ConstIteratorType = typename MapType::ConstIteratorType;
  using IteratorType = typename MapType::IteratorType;

  /// @brief default constructor
  LayoutMap2D() : BaseType() {}

  /// @brief constructor
  LayoutMap2D(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  LayoutMap2D(LayoutMap2D const &rhs) { copy(rhs); }

  /// @brief move constructor
  LayoutMap2D(LayoutMap2D &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  LayoutMap2D &operator=(LayoutMap2D const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  LayoutMap2D &operator=(LayoutMap2D &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for dimensions
  IndexType width() const { return map_.shape(0); }

  /// @brief getter for dimensions
  IndexType height() const { return map_.shape(1); }

  /// @brief setter for the dimensions
  void reshape(IndexType dx, IndexType dy) { map_.reshape(dx, dy); }

  /// @brief getter for element
  ValueType const &operator()(IndexType ix, IndexType iy) const {
    return map_(ix, iy);
  }

  /// @brief getter for element
  ValueType &operator()(IndexType ix, IndexType iy) { return map_(ix, iy); }

  /// @brief getter for element
  ValueType const &at(IndexType ix, IndexType iy) const {
    return map_.at(ix, iy);
  }

  /// @brief getter for element
  ValueType &at(IndexType ix, IndexType iy) { return map_.at(ix, iy); }

  /// @brief getter for element with 1D index
  ValueType const &at(IndexType id) const { return map_.at(id); }

  /// @brief getter for element with 1D index
  ValueType &at(IndexType id) { return map_.at(id); }

  /// @brief iterator
  typename MapType::ConstIteratorType begin() const { return map_.begin(); }

  /// @brief iterator
  typename MapType::IteratorType begin() { return map_.begin(); }

  /// @brief iterator
  typename MapType::ConstIteratorType end() const { return map_.end(); }

  /// @brief iterator
  typename MapType::IteratorType end() { return map_.end(); }

  /// @brief number of elements
  IndexType size() const { return map_.size(); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(LayoutMap2D const &rhs);
  /// @brief move object
  void move(LayoutMap2D &&rhs);
  /// @brief overload output stream
  /// The function cannot be linked if put outside
  friend std::ostream &operator<<(std::ostream &os, LayoutMap2D const &rhs) {
    os << className<decltype(rhs)>() << "("
       << "id_ : " << rhs.id_;
    os << ", map_ : {";
    for (typename LayoutMap2D<T>::IndexType ix = 0; ix < rhs.width(); ++ix) {
      for (typename LayoutMap2D<T>::IndexType iy = 0; iy < rhs.height(); ++iy) {
        os << rhs.map_(ix, iy) << ", ";
      }
      os << "\n";
    }
    os << "}";
    os << ")";
    return os;
  }

  MapType map_;
};

template <typename T> void LayoutMap2D<T>::copy(LayoutMap2D<T> const &rhs) {
  this->BaseType::copy(rhs);
  map_ = rhs.map_;
}

template <typename T> void LayoutMap2D<T>::move(LayoutMap2D<T> &&rhs) {
  this->BaseType::move(std::move(rhs));
  map_ = std::move(rhs.map_);
}

template <typename T>
typename LayoutMap2D<T>::IndexType LayoutMap2D<T>::memory() const {
  return this->BaseType::memory() + sizeof(map_) +
         map_.capacity() * sizeof(typename MapType::ValueType);
}

} // namespace database

OPENPARF_END_NAMESPACE

#endif
