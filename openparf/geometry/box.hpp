/**
 * @file   box.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_GEOMETRY_BOX_HPP_
#define OPENPARF_GEOMETRY_BOX_HPP_

#include "geometry/point.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace geometry {

template <typename CoordinateType>
class Box : public std::array<CoordinateType, 4> {
 public:
  using BaseType = std::array<CoordinateType, 4>;
  using PointType = Point<CoordinateType, 2>;
  using AreaType = typename CoordinateTraits<CoordinateType>::AreaType;

  /// @brief default constructor
  Box() : BaseType() { reset(); }

  /// @brief constructor
  Box(CoordinateType xxl, CoordinateType yyl, CoordinateType xxh,
      CoordinateType yyh)
      : BaseType({xxl, yyl, xxh, yyh}) {}

  /// @brief getter for xl
  CoordinateType const &xl() const { return (*this)[0]; }
  /// @brief getter for yl
  CoordinateType const &yl() const { return (*this)[1]; }
  /// @brief getter for xh
  CoordinateType const &xh() const { return (*this)[2]; }
  /// @brief getter for yh
  CoordinateType const &yh() const { return (*this)[3]; }

  /// @brief width of the box
  CoordinateType width() const { return xh() - xl(); }

  /// @brief height of the box
  CoordinateType height() const { return yh() - yl(); }

  /// @brief reset to invalid box
  void reset() {
    (*this)[0] = (*this)[1] = std::numeric_limits<CoordinateType>::max();
    (*this)[2] = (*this)[3] = std::numeric_limits<CoordinateType>::lowest();
  }

  /// @brief setter for xl
  void setXL(CoordinateType const &v) { (*this)[0] = v; }

  /// @brief setter for yl
  void setYL(CoordinateType const &v) { (*this)[1] = v; }

  /// @brief setter for xh
  void setXH(CoordinateType const &v) { (*this)[2] = v; }

  /// @brief setter for yh
  void setYH(CoordinateType const &v) { (*this)[3] = v; }

  /// @brief set box
  void set(CoordinateType xxl, CoordinateType yyl, CoordinateType xxh,
           CoordinateType yyh) {
    setXL(xxl);
    setYL(yyl);
    setXH(xxh);
    setYH(yyh);
  }

  /// @brief compute the area of the box
  AreaType area() const { return width() * height(); }

  template<typename PointCoordinateType>
  bool contain(Point<PointCoordinateType, 2> const& p) const {
    return xl() <= p.x() && p.x() <= xh()
      && yl() <= p.y() && p.y() <= yh();
  }

  template<typename PointCoordinateType>
  bool contain(PointCoordinateType x, PointCoordinateType y) const {
    return xl() <= x && x <= xh()
      && yl() <= y && y <= yh();
  }


  /// @brief Manhatten distance to a point
  /// TODO: Better type selection. utplacefx generates type in this way:
  /// float float -> float
  /// float int -> float
  /// int int -> int
  template<typename PointCoordinateType>
  Point<PointCoordinateType, 2> manhDist(Point<PointCoordinateType, 2> const& p) const {
    PointCoordinateType xDist = 0.0, yDist = 0.0;
    if (p.x() < xl())       xDist = xl() - p.x();
    else if (p.x() > xh())  xDist = p.x() - xh();
    if (p.y() < yl())       yDist = yl() - p.y();
    else if (p.y() > yh())  yDist = p.y() - yh();
    return Point<PointCoordinateType, 2>(xDist, yDist);
  }


  /// @brief compute the overlap with another box
  template <typename V>
  AreaType overlap(V const &rhs) const {
    auto dx = std::max((CoordinateType)0,
                       std::min(xh(), rhs.xh()) - std::max(xl(), rhs.xl()));
    auto dy = std::max((CoordinateType)0,
                       std::min(yh(), rhs.yh()) - std::max(yl(), rhs.yl()));
    return dx * dy;
  }

  /// @brief check whether a point is within the box
  bool contain(PointType const& p) const {
    return xl() <= p.x() && p.x() <= xh()
      && yl() <= p.y() && p.y() <= yh();
  }

  /// @brief encompass a point
  void encompass(PointType const& p) {
    setXL(std::min(xl(), p.x()));
    setXH(std::max(xh(), p.x()));
    setYL(std::min(yl(), p.y()));
    setYH(std::max(yh(), p.y()));
  }

  /// @brief Extend the box to include the given box
  template <typename V>
  void join(V const &b) {
    setXL(std::min(b.xl(), xl()));
    setYL(std::min(b.yl(), yl()));
    setXH(std::max(b.xh(), xh()));
    setYH(std::max(b.yh(), yh()));
  }

  /// @brief summarize memory usage of the object in bytes
  std::size_t memory() const { return sizeof(*this); }
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Box const &rhs) {
    os << className<decltype(rhs)>() << "(" << rhs.xl() << ", " << rhs.yl()
       << ", " << rhs.xh() << ", " << rhs.yh() << ")";
    return os;
  }
  /// @brief overload input stream
  friend std::istream &operator>>(std::istream &is, Box &rhs) {
    CoordinateType xl, yl, xh, yh;
    is >> rhs.min_corner() >> rhs.max_corner();
    return is;
  }
};

}  // namespace geometry

OPENPARF_END_NAMESPACE

namespace boost {
namespace geometry {
namespace traits {

//////// for box ////////
template <typename CoordinateType>
struct tag<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>>
    : public tag<
          model::box<OPENPARF_NAMESPACE::geometry::Point<CoordinateType, 2>>> {
};

template <typename CoordinateType>
struct point_type<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>> {
  typedef typename OPENPARF_NAMESPACE::geometry::Box<CoordinateType>::PointType
      type;
};

template <typename CoordinateType>
struct indexed_access<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>,
                      min_corner, 0> {
  static inline CoordinateType get(
      OPENPARF_NAMESPACE::geometry::Box<CoordinateType> const &v) {
    return v.xl();
  }
  static inline void set(OPENPARF_NAMESPACE::geometry::Box<CoordinateType> &v,
                         CoordinateType const &value) {
    v.setXL(v, value);
  }
};

template <typename CoordinateType>
struct indexed_access<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>,
                      min_corner, 1> {
  static inline CoordinateType get(
      OPENPARF_NAMESPACE::geometry::Box<CoordinateType> const &v) {
    return v.yl();
  }
  static inline void set(OPENPARF_NAMESPACE::geometry::Box<CoordinateType> &v,
                         CoordinateType const &value) {
    v.setYL(v, value);
  }
};

template <typename CoordinateType>
struct indexed_access<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>,
                      max_corner, 0> {
  static inline CoordinateType get(
      OPENPARF_NAMESPACE::geometry::Box<CoordinateType> const &v) {
    return v.xh();
  }
  static inline void set(OPENPARF_NAMESPACE::geometry::Box<CoordinateType> &v,
                         CoordinateType const &value) {
    v.setXH(v, value);
  }
};

template <typename CoordinateType>
struct indexed_access<OPENPARF_NAMESPACE::geometry::Box<CoordinateType>,
                      max_corner, 1> {
  static inline CoordinateType get(
      OPENPARF_NAMESPACE::geometry::Box<CoordinateType> const &v) {
    return v.yh();
  }
  static inline void set(OPENPARF_NAMESPACE::geometry::Box<CoordinateType> &v,
                         CoordinateType const &value) {
    v.setYH(v, value);
  }
};

}  // namespace traits
}  // namespace geometry
}  // namespace boost

#endif
