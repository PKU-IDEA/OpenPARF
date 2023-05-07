/**
 * @file   half_column_region.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_HALF_COLUMN_REGION_H_
#define OPENPARF_DATABASE_HALF_COLUMN_REGION_H_

#include "database/layout_map.hpp"
#include "geometry/geometry.hpp"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

class HalfColumnRegion : public Object {
public:
  using BaseType = Object;
  using BoxType = geometry::Box<CoordinateType>;

  /// @brief default constructor
  HalfColumnRegion() : BaseType() {}

  /// @brief constructor
  HalfColumnRegion(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  HalfColumnRegion(HalfColumnRegion const &rhs) { copy(rhs); }

  /// @brief move constructor
  HalfColumnRegion(HalfColumnRegion &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  HalfColumnRegion &operator=(HalfColumnRegion const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  HalfColumnRegion &operator=(HalfColumnRegion &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for clock region id
  IndexType clockRegionId() const { return clock_region_id_; }

  /// @brief setter for clock region id
  void setClockRegionId(IndexType id) { clock_region_id_ = id; }

  /// @brief getter for bounding box
  BoxType const &bbox() const { return bbox_; }

  /// @brief setter for bounding box
  void setBbox(BoxType const &v) { bbox_ = v; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(HalfColumnRegion const &rhs);
  /// @brief move object
  void move(HalfColumnRegion &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os,
                                  HalfColumnRegion const &rhs);

  IndexType
      clock_region_id_; ///< the clock region this half column region belongs to
  BoxType bbox_;        ///< bounding box
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
