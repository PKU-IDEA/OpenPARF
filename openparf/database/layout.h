/**
 * @file   layout.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_LAYOUT_H_
#define OPENPARF_DATABASE_LAYOUT_H_

#include "database/clock_region.h"
#include "database/half_column_region.h"
#include "database/resource.h"
#include "database/site.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Layout related data.
class Layout : public Object {
public:
  using BaseType = Object;

  /// @brief default constructor
  Layout() : BaseType() {
    site_map_.setId(0);
    clock_region_map_.setId(0);
    site_type_map_.setId(0);
    resource_map_.setId(0);
  }

  /// @brief constructor
  Layout(IndexType id) : BaseType(id) {
    site_map_.setId(0);
    clock_region_map_.setId(0);
    site_type_map_.setId(0);
    resource_map_.setId(0);
  }

  /// @brief copy constructor
  Layout(Layout const &rhs) { copy(rhs); }

  /// @brief move constructor
  Layout(Layout &&rhs) { move(std::move(rhs)); }

  /// @brief copy assignment
  Layout &operator=(Layout const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Layout &operator=(Layout &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for site map
  SiteMap const &siteMap() const { return site_map_; }

  /// @brief getter for site map
  SiteMap &siteMap() { return site_map_; }

  /// @brief add one site
  Site &addSite(IndexType x, IndexType y, IndexType site_type_id);

  /// @brief getter for clock region map
  ClockRegionMap const &clockRegionMap() const { return clock_region_map_; }

  /// @brief getter for clock region map
  ClockRegionMap &clockRegionMap() { return clock_region_map_; }

  /// @brief number of half column regions
  IndexType numHalfColumnRegions() const { return half_column_regions_.size(); }

  /// @brief getter for half column region
  HalfColumnRegion const &halfColumnRegion(IndexType i) const {
    return half_column_regions_[i];
  }

  /// @brief getter for half column regions
  std::vector<HalfColumnRegion> const &halfColumnRegions() const {
    return half_column_regions_;
  }

  /// @brief add a half column region
  HalfColumnRegion &addHalfColumnRegion(IndexType clock_region_id);

  /// @brief getter for site type map
  SiteTypeMap const &siteTypeMap() const { return site_type_map_; }

  /// @brief getter for site type map
  SiteTypeMap &siteTypeMap() { return site_type_map_; }

  /// @brief getter for a site type given a site
  SiteType const &siteType(Site const &s) const {
    return siteTypeMap().siteType(s.siteTypeId());
  }

  /// @brief getter for a site type given a site
  SiteType &siteType(Site const &s) {
    return siteTypeMap().siteType(s.siteTypeId());
  }

  /// @brief getter for resource map
  ResourceMap const &resourceMap() const { return resource_map_; }

  /// @brief getter for resource map
  ResourceMap &resourceMap() { return resource_map_; }

  /// @brief resize number of sites
  void resizeNumSites(IndexType n) { num_sites_.assign(n, 0); }

  /// @brief getter for number of sites
  IndexType numSites(IndexType site_type_id) const {
    return num_sites_[site_type_id];
  }

  /// @brief setter for number of sites
  void setNumSites(IndexType site_type_id, IndexType v) {
    openparfAssert(site_type_id < num_sites_.size());
    num_sites_[site_type_id] = v;
  }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(Layout const &rhs);
  /// @brief move object
  void move(Layout &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Layout const &rhs);

  SiteMap site_map_;                ///< 2D map for sites
  ClockRegionMap clock_region_map_; ///< 2D map for clock regions
  std::vector<HalfColumnRegion>
      half_column_regions_;   ///< a collection of half column regions
  SiteTypeMap site_type_map_; ///< a collection of site types
  ResourceMap resource_map_;  ///< a collection of resources

  std::vector<IndexType>
      num_sites_; ///< cached number of sites for each site type
                  ///< dimension is the number of site types
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
