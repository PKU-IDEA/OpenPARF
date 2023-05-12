/**
 * @file   site.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_SITE_H_
#define OPENPARF_DATABASE_SITE_H_

#include "container/container.hpp"
#include "database/layout_map.hpp"
#include "database/object.h"
#include "database/site_type.h"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////
/// A brief introduction about FPGA site and resource. The simplest way
/// to describe this problem is that
/// you have a bunch of site types and a bunch of resource types:
/// SiteType1              ResourceType1
/// SiteType2              ResourceType2
/// SiteType3              ResourceType3
/// Each site may contain multiple resource types with each resource
/// type a fixed capacity. For example, SiteType1 may contain
/// 16 ResourceType1 and 8 ResourceType2.
/// In ISPD 2016/2017 contest, site types can be SLICE, FF, DSP, RAM, IO;
/// resource types can be LUT, FF, CARRY8, DSP48E2, RAMB36E2, IO.
/// Different FPGA architectures may have different configurations and
/// names, so it is better to use name mapping to avoid hard coding.
///////////////////////////////////////////////////////////////////////////

namespace database {

/// @brief Site in the layout.
class Site : public Object {
  public:
  using BaseType  = Object;
  using PointType = geometry::Point<IndexType, 2>;
  using BoxType   = geometry::Box<IndexType>;

  /// @brief default constructor
  Site() : BaseType() {
    site_map_id_           = {std::numeric_limits<IndexType>::max(), std::numeric_limits<IndexType>::max()};
    bbox_                  = {std::numeric_limits<IndexType>::max(),
             std::numeric_limits<IndexType>::max(),
             std::numeric_limits<IndexType>::lowest(),
             std::numeric_limits<IndexType>::lowest()};
    clock_region_id_       = std::numeric_limits<IndexType>::max();
    half_column_region_id_ = std::numeric_limits<IndexType>::max();
    site_type_id_          = std::numeric_limits<IndexType>::max();
  }

  /// @brief constructor
  Site(IndexType id) : BaseType(id) {
    site_map_id_           = {std::numeric_limits<IndexType>::max(), std::numeric_limits<IndexType>::max()};
    bbox_                  = {std::numeric_limits<IndexType>::max(),
             std::numeric_limits<IndexType>::max(),
             std::numeric_limits<IndexType>::lowest(),
             std::numeric_limits<IndexType>::lowest()};
    clock_region_id_       = std::numeric_limits<IndexType>::max();
    half_column_region_id_ = std::numeric_limits<IndexType>::max();
    site_type_id_          = std::numeric_limits<IndexType>::max();
  }

  /// @brief copy constructor
  Site(Site const &rhs) { copy(rhs); }

  /// @brief move constructor
  Site(Site &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Site &operator=(Site const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Site &operator=(Site &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief getter for the site map 2D index
  PointType const &siteMapId() const { return site_map_id_; }

  /// @brief setter for the site map 2D index
  void             setSiteMapId(PointType const &v) { site_map_id_ = v; }

  /// @brief getter for the bounding box
  BoxType const &  bbox() const { return bbox_; }

  /// @brief setter for the bounding box
  void             setBbox(BoxType const &v) { bbox_ = v; }

  /// @brief getter for clock region id
  IndexType        clockRegionId() const { return clock_region_id_; }

  /// @brief setter for clock region id
  void             setClockRegionId(IndexType v) { clock_region_id_ = v; }

  /// @brief getter for half column region id
  IndexType        halfColumnRegionId() const { return half_column_region_id_; }

  /// @brief setter for half column region id
  void             setHalfColumnRegionId(IndexType v) { half_column_region_id_ = v; }

  /// @brief getter for type
  IndexType        siteTypeId() const { return site_type_id_; }

  /// @brief setter for type
  void             setSiteTypeId(IndexType v) { site_type_id_ = v; }

  /// @brief summarize memory usage of the object in bytes
  IndexType        memory() const;

  protected:
  /// @brief copy object
  void                 copy(Site const &rhs);
  /// @brief move object
  void                 move(Site &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Site const &rhs);

  PointType            site_map_id_;             ///< 2D index in the site map;
                                                 ///< note that the Object id is the id of valid sites
  BoxType              bbox_;                    ///< The bounding box of the site.
  IndexType            clock_region_id_;         ///< parent clock region index
  IndexType            half_column_region_id_;   ///< parent half column region index
  IndexType            site_type_id_;            ///< index of site type
};

/// @brief Site map for heterogeneous FPGA layout.
/// Note that site map needs to accommodate sites of different sizes.
/// So we introduce a two-level data structure.
/// index_map_ has a dimension of W x H, pointing to the indices of
/// array sites_ which stores the real sites contiguously.
/// If index_map_[x, y] does not correspond to the a site, we set it
/// to infinity.
class SiteMap : public Object {
  public:
  using BaseType          = Object;
  using DimType           = std::array<IndexType, 2>;
  using IteratorType      = typename std::vector<Site>::iterator;
  using ConstIteratorType = typename std::vector<Site>::const_iterator;

  /// @brief default constructor
  SiteMap() : BaseType() {}

  /// @brief constructor
  SiteMap(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  SiteMap(SiteMap const &rhs) { copy(rhs); }

  /// @brief move constructor
  SiteMap(SiteMap &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  SiteMap &operator=(SiteMap const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  SiteMap &operator=(SiteMap &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief shape of the map
  IndexType shape(IndexType dim) const { return dims_[dim]; }

  /// @brief getter for width
  IndexType width() const { return shape(0); }

  /// @brief getter for height
  IndexType height() const { return shape(1); }

  /// @brief setter for the dimensions
  void      reshape(IndexType dx, IndexType dy) {
    dims_ = {dx, dy};
    index_map_.assign(dx * dy, std::numeric_limits<IndexType>::max());
    sites_.clear();
    sites_.reserve(dx * dy);
  }

  /// @brief getter for element
  container::ObserverPtr<const Site> operator()(IndexType ix, IndexType iy) const {
    auto id1d = index1D(ix, iy);
    auto idx  = index_map_[id1d];
    if (idx == std::numeric_limits<IndexType>::max()) {
      return {};
    } else {
      return sites_[idx];
    }
  }

  /// @brief getter for element
  container::ObserverPtr<Site> operator()(IndexType ix, IndexType iy) {
    auto id1d = index1D(ix, iy);
    auto idx  = index_map_[id1d];
    if (idx == std::numeric_limits<IndexType>::max()) {
      return {};
    } else {
      return sites_[idx];
    }
  }

  /// @brief getter for element
  container::ObserverPtr<const Site> at(IndexType ix, IndexType iy) const {
    auto id1d = index1D(ix, iy);
    return at(id1d);
  }

  /// @brief getter for element
  container::ObserverPtr<Site> at(IndexType ix, IndexType iy) {
    auto id1d = index1D(ix, iy);
    return at(id1d);
  }

  /// @brief getter for element with 1D index
  container::ObserverPtr<const Site> at(IndexType id) const {
    auto idx = index_map_.at(id);
    if (idx == std::numeric_limits<IndexType>::max()) {
      return {};
    } else {
      return sites_[idx];
    }
  }

  /// @brief getter for element with 1D index
  container::ObserverPtr<Site> at(IndexType id) {
    auto idx = index_map_.at(id);
    if (idx == std::numeric_limits<IndexType>::max()) {
      return {};
    } else {
      return sites_[idx];
    }
  }

  /// @brief getter for element using exact index
  Site const &atExact(IndexType id) const {
    openparfAssert(id < sites_.size());
    return sites_[id];
  }

  /// @brief getter for element using exact index
  Site &atExact(IndexType id) {
    openparfAssert(id < sites_.size());
    return sites_[id];
  }

  /// @brief getter for site that contains the point x, y
  /// Some site are not 1 by 1 in size, so some integer coordinates might be INSIDE a site,
  /// and at(x, y) would return nothing
  /// This function retrieves a site if the point x, y is inside the site's bounding box.
  container::ObserverPtr<Site> overlapAt(CoordinateType x, CoordinateType y) {
    // TODO: implement a segment tree
    // I want to make it fast while making no assumptions about the shape of the sites
    for (IteratorType s = begin(); s != end(); s++) {
      const auto &bbox = s->bbox();
      if (bbox.xl() <= x && x < bbox.xh() && bbox.yl() <= y && y < bbox.yh()) return container::ObserverPtr<Site>(*s);
    }
    return container::ObserverPtr<Site>();
  }

  container::ObserverPtr<const Site> overlapAt(CoordinateType x, CoordinateType y) const {
    for (ConstIteratorType s = begin(); s != end(); s++) {
      const auto &bbox = s->bbox();
      if (bbox.xl() <= x && x < bbox.xh() && bbox.yl() <= y && y < bbox.yh())
        return container::ObserverPtr<const Site>(*s);
    }
    return container::ObserverPtr<const Site>();
  }


  /// @brief try add a site; if not exist, add the site;
  /// otherwise, return the existing one.
  Site &tryAddSite(IndexType ix, IndexType iy) {
    auto  id1d = index1D(ix, iy);
    auto &idx  = index_map_.at(id1d);
    if (idx == std::numeric_limits<IndexType>::max()) {
      idx = sites_.size();
      sites_.emplace_back();
      auto &site = sites_.back();
      site.setId(idx);
      site.setSiteMapId({ix, iy});
      return site;
    } else {
      return sites_[idx];
    }
  }

  /// @brief compute 1D index
  IndexType         index1D(IndexType ix, IndexType iy) const { return ix * height() + iy; }

  /// @brief iterator only to valid sites
  ConstIteratorType begin() const { return sites_.begin(); }

  /// @brief iterator only to valid sites
  ConstIteratorType end() const { return sites_.end(); }

  /// @brief iterator only to valid sites
  IteratorType      begin() { return sites_.begin(); }

  /// @brief iterator only to valid sites
  IteratorType      end() { return sites_.end(); }

  /// @brief getter for the number of valid sites
  IndexType         size() const { return sites_.size(); }

  /// @brief summarize memory usage of the object in bytes
  IndexType         memory() const;

  protected:
  /// @brief copy object
  void                   copy(SiteMap const &rhs);
  /// @brief move object
  void                   move(SiteMap &&rhs);
  /// @brief overload output stream
  friend std::ostream &  operator<<(std::ostream &os, SiteMap const &rhs);

  DimType                dims_;        ///< dimension
  std::vector<IndexType> index_map_;   ///< 2D index map for sites
  std::vector<Site>      sites_;       ///< record all sites uniquely
};

}   // namespace database

OPENPARF_END_NAMESPACE

#endif
