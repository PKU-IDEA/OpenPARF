/**
 * @file   super_logic_region.h
 * @author Runzhe Tao
 * @date   Nov 2023
 */
#ifndef OPENPARF_DATABASE_SUPER_LOGIC_REGION_H_
#define OPENPARF_DATABASE_SUPER_LOGIC_REGION_H_

#include "database/layout_map.hpp"
#include "geometry/geometry.hpp"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Super logic region
class SuperLogicRegion : public Object {
 public:
  using BaseType  = Object;
  using BoxType   = geometry::Box<CoordinateType>;
  using IndexType = Object::IndexType;

  /// @brief default constructor
  SuperLogicRegion() : BaseType() {}

  /// @brief constructor
  SuperLogicRegion(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  SuperLogicRegion(SuperLogicRegion const &rhs) { copy(rhs); }

  /// @brief move constructor
  SuperLogicRegion(SuperLogicRegion &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  SuperLogicRegion &operator=(SuperLogicRegion const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  SuperLogicRegion &operator=(SuperLogicRegion &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for name
  std::string const &name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &n) { name_ = n; }

  /// @brief getter for type
  std::string const &type() const { return type_; }

  /// @brief setter for name
  void setType(std::string const &t) { type_ = t; }

  /// @brief getter for idx
  IndexType const &idx() const { return idx_; }

  /// @brief setter for idx
  void setIdx(IndexType const &v) { idx_ = v; }

  /// @brief getter for idy
  IndexType const &idy() const { return idy_; }

  /// @brief setter for idy
  void setIdy(IndexType const &v) { idy_ = v; }

  /**
   * @brief getter for the geometry bounding box
   */
  BoxType geometry_bbox() const {
    return {bbox_.xl(), bbox_.yl(), bbox_.xh() + 1, bbox_.yh() + 1};
  }

  /// @brief getter for bounding box
  BoxType const &bbox() const { return bbox_; }

  /// @brief setter for bounding box
  void setBbox(BoxType const &v) { bbox_ = v; }

  /// @brief getter for slr width
  IndexType const &width() const { return width_; }

  /// @brief setter for slr width
  void setWidth(const IndexType &w) { width_ = w; }

  /// @brief getter for slr height
  IndexType const &height() const { return height_; }

  /// @brief setter for slr height
  void setHeight(const IndexType &h) { height_ = h; }

 protected:
  /// @brief copy object
  void copy(SuperLogicRegion const &rhs);
  /// @brief move object
  void move(SuperLogicRegion &&rhs);

  std::string name_;    ///< super logic region name
  std::string type_;    ///< super logic region type
  BoxType     bbox_;    ///< bounding box
  IndexType   idx_;     ///< idx of super logic region
  IndexType   idy_;     ///< idy of super logic region
  IndexType   width_;   ///< width of super logic region
  IndexType   height_;  ///< height of super logic region
};

class SuperLogicRegionMap : public LayoutMap2D<SuperLogicRegion> {
 public:
  using LayoutMap2D<SuperLogicRegion>::LayoutMap2D;

  IndexType slrIdAtLoc(CoordinateTraits<CoordinateType>::AreaType x,
                       CoordinateTraits<CoordinateType>::AreaType y) const {
    const auto      tPoint = geometry::Point<CoordinateTraits<CoordinateType>::AreaType, 2>(x, y);
    const IndexType w      = width();
    const IndexType h      = height();
    for (IndexType slrX = 0; slrX < w; slrX++) {
      for (IndexType slrY = 0; slrY < h; slrY++) {
        auto &slr = at(slrX, slrY);
        if (slr.bbox().contain(tPoint)) {
          return slrX * h + slrY;
        }
      }
    }
    return std::numeric_limits<IndexType>::max();
  }
};

}  // namespace database

OPENPARF_END_NAMESPACE

#endif