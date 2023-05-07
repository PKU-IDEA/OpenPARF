/**
 * @file   site_type.h
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef OPENPARF_DATABASE_SITE_TYPE_H_
#define OPENPARF_DATABASE_SITE_TYPE_H_

#include "container/container.hpp"
#include "database/layout_map.hpp"
#include "database/object.h"
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

/// @brief Types of a site
class SiteType : public Object {
 public:
  using BaseType = Object;

  /// @brief default constructor
  SiteType() : BaseType() {}

  /// @brief constructor
  SiteType(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  SiteType(SiteType const &rhs) { copy(rhs); }

  /// @brief move constructor
  SiteType(SiteType &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  SiteType &operator=(SiteType const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  SiteType &operator=(SiteType &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief getter for name
  std::string name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &v) { name_ = v; }

  /// @brief number of resource types
  IndexType numResources() const { return resource_capacity_.size(); }

  /// @brief set number of resource types
  void setNumResources(IndexType n) { resource_capacity_.assign(n, 0); }

  /// @brief getter for resource capacity
  IndexType resourceCapacity(IndexType resource) const {
    openparfAssert(resource < numResources());
    return resource_capacity_[resource];
  }

  /// @brief setter for resource capacity
  void setResourceCapacity(IndexType resource, IndexType capacity) {
    openparfAssert(resource < numResources());
    resource_capacity_[resource] = capacity;
  }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const override;

 protected:
  /// @brief copy object
  void copy(SiteType const &rhs);
  /// @brief move object
  void move(SiteType &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, SiteType const &rhs);

  std::string name_;  ///< name of the site type
  std::vector<IndexType>
      resource_capacity_;  ///< number of resources available in one site
                           ///< dimension is the same as
                           ///< total number of resources
};

/// @brief A collection of all site types
class SiteTypeMap : public Object {
 public:
  using BaseType = Object;
  using IteratorType = std::vector<SiteType>::iterator;
  using ConstIteratorType = std::vector<SiteType>::const_iterator;

  /// @brief default constructor
  SiteTypeMap() : BaseType() {}

  /// @brief constructor
  SiteTypeMap(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  SiteTypeMap(SiteTypeMap const &rhs) { copy(rhs); }

  /// @brief move constructor
  SiteTypeMap(SiteTypeMap &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  SiteTypeMap &operator=(SiteTypeMap const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  SiteTypeMap &operator=(SiteTypeMap &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief number of site types
  IndexType numSiteTypes() const { return site_types_.size(); }

  /// @brief try add a site type
  SiteType &tryAddSiteType(std::string const &name);

  /// @brief getter for site type
  SiteType const &siteType(IndexType id) const {
    openparfAssert(id < numSiteTypes());
    return site_types_[id];
  }

  /// @brief getter for site type
  SiteType &siteType(IndexType id) { return site_types_[id]; }

  /// @brief getter for a site type index given its name
  IndexType siteTypeId(std::string const &name) const;

  /// @brief getter for a site type given its name
  container::ObserverPtr<const SiteType> siteType(
      std::string const &name) const;

  /// @brief getter for a site type given its name
  container::ObserverPtr<SiteType> siteType(std::string const &name);

  /// @brief iterator for site types
  ConstIteratorType begin() const { return site_types_.begin(); }

  /// @brief iterator for site types
  IteratorType begin() { return site_types_.begin(); }

  /// @brief iterator for site types
  ConstIteratorType end() const { return site_types_.end(); }

  /// @brief iterator for site types
  IteratorType end() { return site_types_.end(); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(SiteTypeMap const &rhs);
  /// @brief move object
  void move(SiteTypeMap &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, SiteTypeMap const &rhs);

  std::vector<SiteType> site_types_;  ///< all site types
  std::unordered_map<std::string, IndexType>
      site_type_name2id_map_;  ///< map site type name to index
};

}  // namespace database

OPENPARF_END_NAMESPACE

#endif
