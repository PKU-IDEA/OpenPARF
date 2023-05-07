/**
 * @file   resource.h
 * @author Yibo Lin
 * @date   Mar 2020
 */
#ifndef OPENPARF_DATABASE_RESOURCE_H_
#define OPENPARF_DATABASE_RESOURCE_H_

#include "container/container.hpp"
#include "database/object.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief a class to represent resource type
class Resource : public Object {
 public:
  using BaseType = Object;

  /// @brief default constructor
  Resource() : BaseType() {}

  /// @brief constructor
  explicit Resource(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  Resource(Resource const &rhs) { copy(rhs); }

  /// @brief move constructor
  Resource(Resource &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Resource &operator=(Resource const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Resource &operator=(Resource &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief getter for name
  std::string name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &v) { name_ = v; }

  /// @brief number of models
  IndexType numModels() const { return model_ids_.size(); }

  /// @brief getter for a model
  IndexType modelId(IndexType id) const { return model_ids_[id]; }

  /// @brief getter for model indices
  std::vector<IndexType> const &modelIds() const { return model_ids_; }

  /// @brief add a model
  void addModel(IndexType id) { model_ids_.push_back(id); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(Resource const &rhs);
  /// @brief move object
  void move(Resource &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Resource const &rhs);

  std::string name_;  ///< name of the resource
  std::vector<IndexType>
      model_ids_;  ///< model indices that can be put in the resource
};

/// @brief a class to record all resources
class ResourceMap : public Object {
 public:
  using BaseType = Object;
  using ConstIteratorType = std::vector<Resource>::const_iterator;
  using IteratorType = std::vector<Resource>::iterator;

  /// @brief default constructor
  ResourceMap() : BaseType() {}

  /// @brief constructor
  explicit ResourceMap(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  ResourceMap(ResourceMap const &rhs) { copy(rhs); }

  /// @brief move constructor
  ResourceMap(ResourceMap &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  ResourceMap &operator=(ResourceMap const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  ResourceMap &operator=(ResourceMap &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief number of resources
  IndexType numResources() const { return resources_.size(); }

  /// @brief getter for resource
  Resource const &resource(IndexType id) const {
    openparfAssert(id < numResources());
    return resources_[id];
  }

  /// @brief getter for resource
  Resource &resource(IndexType id) {
    openparfAssert(id < numResources());
    return resources_[id];
  }

  /// @brief add a resource
  Resource &tryAddResource(std::string const &name);

  /// @brief add a model cell to resource
  Resource &tryAddModelResource(IndexType model_id,
                                std::string const &resource_name);

  /// @brief number of models recorded
  IndexType numModels() const { return model2resource_ids_.size(); }

  /// @brief set number of models
  void setNumModels(IndexType v) { model2resource_ids_.resize(v); }

  /// @brief getter for resource given resource name
  IndexType resourceId(std::string const &name) const;

  /// @brief getter for resource given resource name
  container::ObserverPtr<const Resource> resource(
      std::string const &name) const;

  /// @brief getter for resource given resource name
  container::ObserverPtr<Resource> resource(std::string const &name);

  /// @brief query the resource for a cell model
  std::vector<IndexType> const &modelResourceIds(IndexType model_id) const;

  /// @brief iterator for resources
  ConstIteratorType begin() const { return resources_.begin(); }

  /// @brief iterator for resources
  ConstIteratorType end() const { return resources_.end(); }

  /// @brief iterator for resources
  IteratorType begin() { return resources_.begin(); }

  /// @brief iterator for resources
  IteratorType end() { return resources_.end(); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(ResourceMap const &rhs);
  /// @brief move object
  void move(ResourceMap &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, ResourceMap const &rhs);

  std::vector<Resource> resources_;  ///< all resources
  std::unordered_map<std::string, IndexType>
      resource_name2id_map_;  ///< map resource name to index
  std::vector<std::vector<IndexType>> model2resource_ids_;
  ///< map model name to resource index
};

}  // namespace database

OPENPARF_END_NAMESPACE

#endif
