/**
 * @file   clock_region.h
 * @author Yibo Lin
 * @date   Mar 2020
 */
#ifndef OPENPARF_DATABASE_CLOCK_REGION_H_
#define OPENPARF_DATABASE_CLOCK_REGION_H_

#include "database/layout_map.hpp"
#include "geometry/geometry.hpp"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Clock region
class ClockRegion : public Object {
public:
    using BaseType = Object;
    using BoxType  = geometry::Box<CoordinateType>;

    /// @brief default constructor
    ClockRegion() : BaseType() {}

    /// @brief constructor
    ClockRegion(IndexType id) : BaseType(id) {}

    /// @brief copy constructor
    ClockRegion(ClockRegion const &rhs) { copy(rhs); }

    /// @brief move constructor
    ClockRegion(ClockRegion &&rhs) noexcept { move(std::move(rhs)); }

    /// @brief copy assignment
    ClockRegion &operator=(ClockRegion const &rhs) {
        if (this != &rhs) { copy(rhs); }
        return *this;
    }

    /// @brief move assignment
    ClockRegion &operator=(ClockRegion &&rhs) noexcept {
        if (this != &rhs) { move(std::move(std::move(rhs))); }
        return *this;
    }

    /// @brief getter for name
    std::string const &name() const { return name_; }

    /// @brief setter for name
    void setName(std::string const &n) { name_ = n; }

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

    /// @brief number of half column regions
    IndexType numHalfColumnRegions() const { return half_column_region_ids_.size(); }

    /// @brief getter for half column region
    IndexType halfColumnRegionId(IndexType i) const { return half_column_region_ids_[i]; }

    /// @brief getter for half column regions
    std::vector<IndexType> const &halfColumnRegionIds() const { return half_column_region_ids_; }

    /// @brief add a half column region
    void addHalfColumnRegion(IndexType id) { half_column_region_ids_.push_back(id); }

    /// @brief set the left boundary of half columns within this clock region.
    void setHcXMin(IndexType hc_xmin) { hc_xmin_ = hc_xmin; }

    ///@brief The left boundary of half columns within this clock regions.
    IndexType const &HcXMin() const { return hc_xmin_; }

    /// @brief set the right boundary of half columns within this clock region.
    void setHcXMax(IndexType hc_xmax) { hc_xmax_ = hc_xmax; }

    ///@brief The right boundary of half columns within this clock regions.
    IndexType const &HcXMAx() const { return hc_xmax_; }

    /// @brief resize number of sites
    void resizeNumSites(IndexType n) { num_sites_.assign(n, 0); }

    /// @brief getter for number of sites
    IndexType numSites(IndexType site_type_id) const { return num_sites_[site_type_id]; }

    /// @brief setter for number of sites
    void setNumSites(IndexType site_type_id, IndexType v) {
        openparfAssert(site_type_id < num_sites_.size());
        num_sites_[site_type_id] = v;
    }

    /// @brief add one site to a specific type
    IndexType incrNumSites(IndexType site_type_id, IndexType val) {
        openparfAssert(site_type_id < num_sites_.size());
        return (num_sites_[site_type_id] += val);
    }

    /// @brief summarize memory usage of the object in bytes
    IndexType memory() const;

protected:
    /// @brief copy object
    void copy(ClockRegion const &rhs);
    /// @brief move object
    void move(ClockRegion &&rhs);
    /// @brief overload output stream
    friend std::ostream &operator<<(std::ostream &os, ClockRegion const &rhs);

    std::string name_;      ///< clock region name
    BoxType     bbox_;      ///< bounding box
    IndexType   hc_xmin_;   ///< left boundary of half_columns with in this clock region.
    IndexType   hc_xmax_;   ///< left boundary of half_columns with in this clock region.

    std::vector<IndexType> half_column_region_ids_;   ///< half column region indices in the clock
                                                      ///< region
    std::vector<IndexType> num_sites_;                ///< cached number of sites for each site type
                                                      ///< dimension is the number of site types
};

class ClockRegionMap : public LayoutMap2D<ClockRegion> {
public:
    using LayoutMap2D<ClockRegion>::LayoutMap2D;
    // TODO: implement finding in segment tree
    // Also TODO: add an id field to class ClockRegion
    // Also TODO: 2d to 1d conversion for LayoutMap2D
    IndexType crIdAtLoc(CoordinateTraits<CoordinateType>::AreaType x,
                        CoordinateTraits<CoordinateType>::AreaType y) const {
        for (IndexType crX = 0; crX < width(); crX++) {
            for (IndexType crY = 0; crY < height(); crY++) {
                auto &cr = at(crX, crY);
                if (cr.bbox().contain(
                            geometry::Point<CoordinateTraits<CoordinateType>::AreaType, 2>(x, y))) {
                    return crX * height() + crY;
                }
            }
        }
        return std::numeric_limits<IndexType>::max();
    }
};

}   // namespace database

OPENPARF_END_NAMESPACE

#endif
