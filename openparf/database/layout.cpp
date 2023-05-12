/**
 * @file   layout.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/layout.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Layout::copy(Layout const &rhs) {
  this->BaseType::copy(rhs);
  site_map_ = rhs.site_map_;
  clock_region_map_ = rhs.clock_region_map_;
  half_column_regions_ = rhs.half_column_regions_;
  site_type_map_ = rhs.site_type_map_;
  resource_map_ = rhs.resource_map_;
  num_sites_ = rhs.num_sites_;
}

void Layout::move(Layout &&rhs) {
  this->BaseType::move(std::move(rhs));
  site_map_ = std::move(site_map_);
  clock_region_map_ = std::move(rhs.clock_region_map_);
  half_column_regions_ = std::move(rhs.half_column_regions_);
  site_type_map_ = std::move(rhs.site_type_map_);
  resource_map_ = std::move(rhs.resource_map_);
  num_sites_ = std::move(rhs.num_sites_);
}

Site &Layout::addSite(Layout::IndexType x, Layout::IndexType y,
                      Layout::IndexType site_type_id) {
  auto &s = site_map_.tryAddSite(x, y);
  s.setSiteTypeId(site_type_id);
  ++num_sites_[site_type_id];
  return s;
}

HalfColumnRegion &
Layout::addHalfColumnRegion(Layout::IndexType clock_region_id) {
  half_column_regions_.emplace_back(half_column_regions_.size());
  auto &hc = half_column_regions_.back();
  hc.setClockRegionId(clock_region_id);
  clock_region_map_.at(clock_region_id).addHalfColumnRegion(hc.id());
  return hc;
}

Layout::IndexType Layout::memory() const {
  return this->BaseType::memory() + site_map_.memory() +
         clock_region_map_.memory() +
         half_column_regions_.capacity() *
             sizeof(decltype(half_column_regions_)::value_type) +
         resource_map_.memory() + sizeof(num_sites_);
}

std::ostream &operator<<(std::ostream &os, Layout const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", \nsite_map_ : " << rhs.site_map_
     << ", \nclock_region_map_ : " << rhs.clock_region_map_
     << ", \nhalf_column_regions_ : " << rhs.half_column_regions_
     << ", \nsite_type_map_ : " << rhs.site_type_map_
     << ", \nresource_map_ : " << rhs.resource_map_ << ", \nnum_sites_ : (";
  const char *delimiter = "";
  for (auto v : rhs.num_sites_) {
    os << delimiter << v;
    delimiter = ", ";
  }
  os << " )";
  os << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
