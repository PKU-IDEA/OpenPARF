/**
 * @file   site.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/site.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Site::copy(Site const &rhs) {
  this->BaseType::copy(rhs);
  site_map_id_ = rhs.site_map_id_;
  bbox_ = rhs.bbox_;
  clock_region_id_ = rhs.clock_region_id_;
  half_column_region_id_ = rhs.half_column_region_id_;
  site_type_id_ = rhs.site_type_id_;
}

void Site::move(Site &&rhs) {
  this->BaseType::move(std::move(rhs));
  site_map_id_ = std::move(rhs.site_map_id_);
  bbox_ = std::move(rhs.bbox_);
  clock_region_id_ = std::move(rhs.clock_region_id_);
  half_column_region_id_ = std::move(rhs.half_column_region_id_);
  site_type_id_ = std::move(rhs.site_type_id_);
}

Site::IndexType Site::memory() const {
  return this->BaseType::memory() + sizeof(site_map_id_) + sizeof(bbox_) +
         sizeof(clock_region_id_) + sizeof(half_column_region_id_) +
         sizeof(site_type_id_);
}

std::ostream &operator<<(std::ostream &os, Site const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", bbox_ : " << rhs.bbox_
     << ", site_map_id_ : " << rhs.site_map_id_
     << ", clock_region_id_ : " << rhs.clock_region_id_
     << ", half_column_region_id_ : " << rhs.half_column_region_id_
     << ", site_type_id_ : " << rhs.site_type_id_ << ")";
  return os;
}

void SiteMap::copy(SiteMap const &rhs) {
  this->BaseType::copy(rhs);
  dims_ = rhs.dims_;
  index_map_ = rhs.index_map_;
  sites_ = rhs.sites_;
}

void SiteMap::move(SiteMap &&rhs) {
  this->BaseType::move(std::move(rhs));
  dims_ = std::move(rhs.dims_);
  index_map_ = std::move(index_map_);
  sites_ = std::move(rhs.sites_);
}

SiteMap::IndexType SiteMap::memory() const {
  return this->BaseType::memory() + sizeof(dims_) + sizeof(index_map_) +
         sizeof(sites_) + index_map_.capacity() * sizeof(IndexType) +
         sites_.capacity() * sizeof(Site);
}

std::ostream &operator<<(std::ostream &os, SiteMap const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", dims_ : (" << rhs.width() << ", " << rhs.height() << ")";
  os << ", sites_ : [";
  const char *delimiter = "";
  for (auto const &v : rhs.sites_) {
    os << delimiter << v;
    delimiter = ", ";
  }
  os << "]";
  os << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
