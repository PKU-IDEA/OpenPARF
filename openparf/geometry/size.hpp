/**
 * @file   size.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_GEOMETRY_SIZE_HPP_
#define OPENPARF_GEOMETRY_SIZE_HPP_

#include "util/util.h"
#include <boost/geometry.hpp>
#include <numeric>
#include <sstream>

OPENPARF_BEGIN_NAMESPACE

namespace geometry {

template <typename CoordinateType, std::size_t DimensionCount>
class BaseSize : public std::array<CoordinateType, DimensionCount> {
public:
  using BaseType = std::array<CoordinateType, DimensionCount>;

  /// @brief default constructor
  BaseSize() : BaseType() {
    this->fill(std::numeric_limits<CoordinateType>::max());
  }
  /// @brief constructor
  template <class... Args>
  BaseSize(CoordinateType const &first, Args... args)
      : BaseType({first, args...}) {}
  /// @brief copy constructor
  BaseSize(BaseSize const &rhs) : BaseType(rhs) {}
  /// @brief copy assignment
  BaseSize &operator=(BaseSize const &rhs) {
    if (this != &rhs) {
      std::copy(rhs.begin(), rhs.end(), this->begin());
    }
    return *this;
  }

  /// @brief getter for data
  template <std::size_t Dimension> CoordinateType const &get() const {
    return (*this)[Dimension];
  }
  /// @brief setter for data
  template <std::size_t Dimension> void set(CoordinateType const &v) {
    (*this)[Dimension] = v;
  }
  /// @brief setter for data
  void set(std::initializer_list<CoordinateType> rhs) { *this = rhs; }

  /// @brief getter for the product
  CoordinateType product() const {
    return std::accumulate(this->begin(), this->end(), (CoordinateType)1,
                           std::multiplies<CoordinateType>());
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, BaseSize const &rhs) {
    os << className<decltype(rhs)>() << "(";
    for (std::size_t i = 0; i < DimensionCount; ++i) {
      if (i) {
        os << ", ";
      }
      os << rhs.at(i);
    }
    os << ")";
    return os;
  }
};

template <typename CoordinateType, std::size_t DimensionCount> class Size;

template <typename CoordinateType>
class Size<CoordinateType, 2> : public BaseSize<CoordinateType, 2> {
public:
  using BaseType = BaseSize<CoordinateType, 2>;

  /// @brief default constructor
  Size() : BaseType() {}
  /// @brief constructor
  Size(CoordinateType const &v0, CoordinateType const &v1) : BaseType(v0, v1) {}
  /// @brief copy constructor
  Size(Size const &rhs) : BaseType(rhs) {}
  /// @brief copy assignment
  Size &operator=(Size const &rhs) {
    if (this != &rhs) {
      this->BaseType::operator=(rhs);
    }
    return *this;
  }

  /// @brief getter for x
  CoordinateType const &x() const { return this->BaseType::template get<0>(); }
  /// @brief getter for width
  CoordinateType const &width() const { return x(); }
  /// @brief getter for y
  CoordinateType const &y() const { return this->BaseType::template get<1>(); }
  /// @brief getter for height
  CoordinateType const &height() const { return y(); }
  /// @brief setter for x
  void setX(CoordinateType const &v) { this->BaseType::template set<0>(v); }
  /// @brief setter for width
  void setWidth(CoordinateType const &v) { setX(v); }
  /// @brief setter for y
  void setY(CoordinateType const &v) { this->BaseType::template set<1>(v); }
  /// @brief setter for height
  void setHeight(CoordinateType const &v) { setY(v); }
  /// @brief setter for x and y
  void set(CoordinateType const &v0, CoordinateType const &v1) {
    this->setX(v0);
    this->setY(v1);
  }

  /// @brief getter for the product
  CoordinateType product() const {
    return this->BaseType::product();
  }

  /// @brief getter for the area
  CoordinateType area() const {
    return product();
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Size const &rhs) {
    os << className<decltype(rhs)>() << "(" << rhs.x() << ", " << rhs.y()
       << ")";
    return os;
  }
};

template <typename CoordinateType>
class Size<CoordinateType, 3> : public BaseSize<CoordinateType, 3> {
public:
  using BaseType = BaseSize<CoordinateType, 3>;

  /// @brief default constructor
  Size() : BaseType() {}
  /// @brief constructor
  Size(CoordinateType const &v0, CoordinateType const &v1,
       CoordinateType const &v2)
      : BaseType(v0, v1, v2) {}
  /// @brief copy constructor
  Size(Size const &rhs) : BaseType(rhs) {}
  /// @brief copy assignment
  Size &operator=(Size const &rhs) {
    if (this != &rhs) {
      this->BaseType::operator=(rhs);
    }
    return *this;
  }

  /// @brief getter for x
  CoordinateType const &x() const { return this->BaseType::template get<0>(); }
  /// @brief getter for y
  CoordinateType const &y() const { return this->BaseType::template get<1>(); }
  /// @brief getter for z
  CoordinateType const &z() const { return this->BaseType::template get<2>(); }
  /// @brief setter for x
  void setX(CoordinateType const &v) { this->BaseType::template set<0>(v); }
  /// @brief setter for y
  void setY(CoordinateType const &v) { this->BaseType::template set<1>(v); }
  /// @brief setter for z
  void setZ(CoordinateType const &v) { this->BaseType::template set<2>(v); }
  /// @brief setter for x, y, z
  void set(CoordinateType const &v0, CoordinateType const &v1,
           CoordinateType const &v2) {
    this->setX(v0);
    this->setY(v1);
    this->setZ(v2);
  }

  /// @brief getter for the product
  CoordinateType product() const {
    return this->BaseType::product();
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Size const &rhs) {
    os << className<decltype(rhs)>() << "(" << rhs.x() << ", " << rhs.y()
       << ", " << rhs.z() << ")";
    return os;
  }
};

} // namespace geometry

OPENPARF_END_NAMESPACE

#endif
