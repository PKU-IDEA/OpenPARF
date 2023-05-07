/**
 * @file   site_type.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 */

#include "database/site_type.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void SiteType::copy(SiteType const &rhs) {
  this->BaseType::copy(rhs);
  name_ = rhs.name_;
  resource_capacity_ = rhs.resource_capacity_;
}

void SiteType::move(SiteType &&rhs) {
  this->BaseType::move(std::move(rhs));
  name_ = std::move(rhs.name_);
  resource_capacity_ = std::move(rhs.resource_capacity_);
}

SiteType::IndexType SiteType::memory() const {
  return this->BaseType::memory() + sizeof(name_) + sizeof(resource_capacity_) +
         name_.size() * sizeof(char) +
         resource_capacity_.capacity() *
             sizeof(decltype(resource_capacity_)::value_type);
}

std::ostream &operator<<(std::ostream &os, SiteType const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", name_ : " << rhs.name_
     << ", resource_capacity_ : " << rhs.resource_capacity_ << ")";
  return os;
}

void SiteTypeMap::copy(SiteTypeMap const &rhs) {
  this->BaseType::copy(rhs);
  site_types_ = rhs.site_types_;
  site_type_name2id_map_ = rhs.site_type_name2id_map_;
}

void SiteTypeMap::move(SiteTypeMap &&rhs) {
  this->BaseType::move(std::move(rhs));
  site_types_ = std::move(rhs.site_types_);
  site_type_name2id_map_ = std::move(rhs.site_type_name2id_map_);
}

SiteTypeMap::IndexType SiteTypeMap::memory() const {
  IndexType ret = this->BaseType::memory() + sizeof(site_types_) +
                  sizeof(site_type_name2id_map_) +
                  site_type_name2id_map_.size() * sizeof(std::string);
  for (auto const &v : site_types_) {
    ret += v.memory();
  }
  return ret;
}

std::ostream &operator<<(std::ostream &os, SiteTypeMap const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", site_types_ : " << rhs.site_types_;
  os << ", site_type_name2id_map_ : {";
  const char *delimiter = "";
  for (auto const &kvp : rhs.site_type_name2id_map_) {
    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
    delimiter = ", ";
  }
  os << "}";
  os << ")";
  return os;
}

SiteType &SiteTypeMap::tryAddSiteType(std::string const &name) {
  auto ret = site_type_name2id_map_.find(name);
  if (ret == site_type_name2id_map_.end()) {
    site_type_name2id_map_.emplace(name, site_types_.size());
    site_types_.emplace_back(site_types_.size());
    auto &st = site_types_.back();
    st.setName(name);
    return st;
  } else {
    return site_types_[ret->second];
  }
}

SiteTypeMap::IndexType SiteTypeMap::siteTypeId(std::string const &name) const {
  auto ret = site_type_name2id_map_.find(name);
  if (ret == site_type_name2id_map_.end()) {
    return std::numeric_limits<IndexType>::max();
  } else {
    return ret->second;
  }
}

container::ObserverPtr<const SiteType> SiteTypeMap::siteType(
    std::string const &name) const {
  auto ret = siteTypeId(name);
  if (ret == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return &siteType(ret);
  }
}

container::ObserverPtr<SiteType> SiteTypeMap::siteType(
    std::string const &name) {
  auto ret = siteTypeId(name);
  if (ret == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return &siteType(ret);
  }
}

}  // namespace database

OPENPARF_END_NAMESPACE
