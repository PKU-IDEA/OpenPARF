/**
 * @file   geometry.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_PYBIND_GEOMETRY_H_
#define OPENPARF_PYBIND_GEOMETRY_H_

#include "geometry/geometry.hpp"
#include "pybind/util.h"

namespace geometry = OPENPARF_NAMESPACE::geometry;

using Point2DInt = geometry::Point<int32_t, 2>;
using VectorPoint2DInt = std::vector<Point2DInt>;
using Point2DUint = geometry::Point<uint32_t, 2>;
using VectorPoint2DUint = std::vector<Point2DUint>;
using Point2DFloat = geometry::Point<float, 2>;
using VectorPoint2DFloat = std::vector<Point2DFloat>;
using Point2DDouble = geometry::Point<double, 2>;
using VectorPoint2DDouble = std::vector<Point2DDouble>;
using Point3DInt = geometry::Point<int32_t, 3>;
using VectorPoint3DInt = std::vector<Point3DInt>;
using Point3DUint = geometry::Point<uint32_t, 3>;
using VectorPoint3DUint = std::vector<Point3DUint>;
using Point3DFloat = geometry::Point<float, 3>;
using VectorPoint3DFloat = std::vector<Point3DFloat>;
using Point3DDouble = geometry::Point<double, 3>;
using VectorPoint3DDouble = std::vector<Point3DDouble>;
using BoxInt = geometry::Box<int32_t>;
using VectorBoxInt = std::vector<BoxInt>;
using BoxUint = geometry::Box<uint32_t>;
using VectorBoxUint = std::vector<BoxUint>;
using BoxFloat = geometry::Box<float>;
using VectorBoxFloat = std::vector<BoxFloat>;
using BoxDouble = geometry::Box<double>;
using VectorBoxDouble = std::vector<BoxDouble>;
using Size2DInt = geometry::Size<int32_t, 2>;
using VectorSize2DInt = std::vector<Size2DInt>;
using Size2DUint = geometry::Size<uint32_t, 2>;
using VectorSize2DUint = std::vector<Size2DUint>;
using Size2DFloat = geometry::Size<float, 2>;
using VectorSize2DFloat = std::vector<Size2DFloat>;
using Size2DDouble = geometry::Size<double, 2>;
using VectorSize2DDouble = std::vector<Size2DDouble>;
using Size3DInt = geometry::Size<int32_t, 3>;
using VectorSize3DInt = std::vector<Size3DInt>;
using Size3DUint = geometry::Size<uint32_t, 3>;
using VectorSize3DUint = std::vector<Size3DUint>;
using Size3DFloat = geometry::Size<float, 3>;
using VectorSize3DFloat = std::vector<Size3DFloat>;
using Size3DDouble = geometry::Size<double, 3>;
using VectorSize3DDouble = std::vector<Size3DDouble>;

/// Cast shapes to array
namespace pybind11 {
namespace detail {

// template <typename T, std::size_t DimensionCount>
// struct type_caster<geometry::Point<T, DimensionCount>>
//    : array_caster<geometry::Point<T, DimensionCount>, T, false,
//                   DimensionCount> {};

// template <typename T, std::size_t DimensionCount>
// struct type_caster<geometry::Size<T, DimensionCount>>
//    : array_caster<geometry::Size<T, DimensionCount>, T, false,
//                   DimensionCount> {};

// template <typename T>
// struct type_caster<geometry::Box<T>>
//    : array_caster<geometry::Box<T>, T, false, 4> {};

}  // namespace detail
}  // namespace pybind11

#endif
