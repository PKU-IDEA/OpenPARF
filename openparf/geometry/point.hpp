/**
 * @file   point.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_GEOMETRY_POINT_HPP_
#define OPENPARF_GEOMETRY_POINT_HPP_

#include <boost/geometry.hpp>
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace geometry {

/// @brief Point in cartesian coordinate system
template <typename CoordinateType, std::size_t DimensionCount>
class BasePoint : public std::array<CoordinateType, DimensionCount> {
 public:
  using BaseType = std::array<CoordinateType, DimensionCount>;

  /// @brief default constructor
  BasePoint() : BaseType() {
    this->fill(std::numeric_limits<CoordinateType>::max());
  }
  /// @brief constructor
  template <class... Args>
  BasePoint(CoordinateType const &first, Args... args)
      : BaseType({first, args...}) {}
  /// @brief copy constructor
  BasePoint(BasePoint const &rhs) : BaseType(rhs) {}
  /// @brief copy assignment
  BasePoint &operator=(BasePoint const &rhs) {
    if (this != &rhs) {
      std::copy(rhs.begin(), rhs.end(), this->begin());
    }
    return *this;
  }

  /// @brief getter for data
  template <std::size_t Dimension>
  CoordinateType const &get() const {
    return (*this)[Dimension];
  }
  /// @brief setter for data
  template <std::size_t Dimension>
  void set(CoordinateType const &v) {
    (*this)[Dimension] = v;
  }
  /// @brief setter for data
  void set(std::initializer_list<CoordinateType> rhs) { *this = rhs; }

  /// @brief summarize memory usage of the object in bytes
  std::size_t memory() const { return sizeof(*this); }
};

template <typename CoordinateType, std::size_t DimensionCount>
class Point;

template <typename CoordinateType>
class Point<CoordinateType, 2> : public BasePoint<CoordinateType, 2> {
 public:
  using BaseType = BasePoint<CoordinateType, 2>;
  /// @brief default constructor
  Point() : BaseType() {
    setX(std::numeric_limits<CoordinateType>::max());
    setY(std::numeric_limits<CoordinateType>::max());
  }
  /// @brief constructor
  Point(CoordinateType const &v0, CoordinateType const &v1) {
    setX(v0);
    setY(v1);
  }
  /// @brief copy constructor
  Point(Point const &rhs) : BaseType(rhs) {}

  /// @brief getter for x
  CoordinateType const &x() const { return this->get<0>(); }
  /// @brief getter for y
  CoordinateType const &y() const { return this->get<1>(); }
  /// @brief getter general
  template <std::size_t Dimension>
  CoordinateType const &get() const {
    return this->BaseType::template get<Dimension>();
  }

  Point    left() const     { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); return Point(this->get<0>() - 1, this->get<1>()); }
  Point    right() const    { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); return Point(this->get<0>() + 1, this->get<1>()); }
  Point    bottom() const   { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); return Point(this->get<0>(), this->get<1>() - 1); }
  Point    top() const      { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); return Point(this->get<0>(), this->get<1>() + 1); }

  void    toLeft()      { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); this->set<0>(x()-1); }
  void    toRight()     { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); this->set<0>(x()+1); }
  void    toBottom()    { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); this->set<1>(y()-1); }
  void    toTop()       { static_assert(std::is_integral<CoordinateType>::value, "Integral type is required."); this->set<1>(y()+1); }



  /// @brief setter for x
  void setX(CoordinateType const &v) { this->set<0>(v); }
  /// @brief setter for y
  void setY(CoordinateType const &v) { this->set<1>(v); }
  /// @brief setter general
  template <std::size_t Dimension>
  void set(CoordinateType const &v) {
    this->BaseType::template set<Dimension>(v);
  }
  /// @brief setter for x and y
  void set(CoordinateType const &v0, CoordinateType const &v1) {
    this->setX(v0);
    this->setY(v1);
  }

  /// @brief manhattan distance
  CoordinateType manhattanDistance(Point const& rhs) const {
    return std::abs(x() - rhs.x()) + std::abs(y() - rhs.y());
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Point const &rhs) {
    os << className<decltype(rhs)>() << "(" << rhs.x() << ", " << rhs.y()
       << ")";
    return os;
  }
  /// @brief overload input stream
  friend std::istream &operator>>(std::istream &is, Point &rhs) {
    CoordinateType x, y;
    is >> x >> y;
    rhs.set(x, y);
    return is;
  }
  /// @brief overalod operator+
  friend Point operator+(Point const& v1, Point const& v2) {
    return Point(v1.x() + v2.x(), v1.y() + v2.y());
  }
  /// @brief overalod operator-
  friend Point operator-(Point const& v1, Point const& v2) {
    return Point(v1.x() - v2.x(), v1.y() - v2.y());
  }
};

template <typename CoordinateType>
class Point<CoordinateType, 3> : public BasePoint<CoordinateType, 3> {
 public:
  using BaseType = BasePoint<CoordinateType, 3>;
  /// @brief default constructor
  Point() : BaseType() {
    setX(std::numeric_limits<CoordinateType>::max());
    setY(std::numeric_limits<CoordinateType>::max());
    setZ(std::numeric_limits<CoordinateType>::max());
  }
  /// @brief constructor
  Point(CoordinateType const &v0, CoordinateType const &v1,
        CoordinateType const &v2) {
    setX(v0);
    setY(v1);
    setZ(v2);
  }
  /// @brief copy constructor
  Point(Point const &rhs) : BaseType(rhs) {}

  /// @brief getter for x
  CoordinateType const &x() const { return this->get<0>(); }
  /// @brief getter for y
  CoordinateType const &y() const { return this->get<1>(); }
  /// @brief getter for z
  CoordinateType const &z() const { return this->get<2>(); }
  /// @brief getter general
  template <std::size_t Dimension>
  CoordinateType const &get() const {
    return this->BaseType::template get<Dimension>();
  }
  /// @brief setter for x
  void setX(CoordinateType const &v) { this->set<0>(v); }
  /// @brief setter for y
  void setY(CoordinateType const &v) { this->set<1>(v); }
  /// @brief setter for z
  void setZ(CoordinateType const &v) { this->set<2>(v); }
  /// @brief setter general
  template <std::size_t Dimension>
  void set(CoordinateType const &v) {
    this->BaseType::template set<Dimension>(v);
  }
  /// @brief setter for x, y, z
  void set(CoordinateType const &v0, CoordinateType const &v1,
           CoordinateType const &v2) {
    this->setX(v0);
    this->setY(v1);
    this->setZ(v2);
  }

  /// @brief manhattan distance
  CoordinateType manhattanDistance(Point const& rhs) const {
    return std::abs(x() - rhs.x()) + std::abs(y() - rhs.y()) + std::abs(z() - rhs.z());
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Point const &rhs) {
    os << className<decltype(rhs)>() << "(" << rhs.x() << ", " << rhs.y()
       << ", " << rhs.z() << ")";
    return os;
  }
  /// @brief overload input stream
  friend std::istream &operator>>(std::istream &is, Point &rhs) {
    CoordinateType x, y, z;
    is >> x >> y >> z;
    rhs.set(x, y, z);
    return is;
  }
  /// @brief overalod operator+
  friend Point operator+(Point const& v1, Point const& v2) {
    return Point(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
  }
  /// @brief overalod operator-
  friend Point operator-(Point const& v1, Point const& v2) {
    return Point(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
  }
};

}  // namespace geometry

OPENPARF_END_NAMESPACE

namespace boost {
namespace geometry {
namespace traits {

//////// for point ////////
template <typename CoordinateType, std::size_t DimensionCount>
struct tag<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>> {
  typedef point_tag type;
};

template <typename CoordinateType, std::size_t DimensionCount>
struct point_type<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>> {
  typedef typename OPENPARF_NAMESPACE::geometry::Point<CoordinateType,
                                                       DimensionCount>
      type;
};

template <typename CoordinateType, std::size_t DimensionCount>
struct coordinate_type<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>> {
  typedef CoordinateType type;
};

template <typename CoordinateType, std::size_t DimensionCount>
struct coordinate_system<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>>
    : public coordinate_system<
          model::point<CoordinateType, DimensionCount, cs::cartesian>> {};

template <typename CoordinateType, std::size_t DimensionCount>
struct dimension<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>>
    : public dimension<
          model::point<CoordinateType, DimensionCount, cs::cartesian>> {};

template <typename CoordinateType, std::size_t DimensionCount,
          std::size_t Dimension>
struct access<
    OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount>,
    Dimension> {
  static inline CoordinateType get(
      OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount> const
          &p) {
    return p.template get<Dimension>();
  }

  static inline void set(
      OPENPARF_NAMESPACE::geometry::Point<CoordinateType, DimensionCount> &p,
      CoordinateType const &value) {
    p.template set<Dimension>(value);
  }
};

}  // namespace traits
}  // namespace geometry
}  // namespace boost

#endif
