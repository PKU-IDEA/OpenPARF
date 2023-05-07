/**
 * @file   resource.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/resource.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Resource::copy(Resource const &rhs) {
  this->BaseType::copy(rhs);
  name_ = rhs.name_;
  model_ids_ = rhs.model_ids_;
}

void Resource::move(Resource &&rhs) {
  this->BaseType::move(std::move(rhs));
  name_ = std::move(rhs.name_);
  model_ids_ = std::move(rhs.model_ids_);
}

Resource::IndexType Resource::memory() const {
  return this->BaseType::memory() + sizeof(name_) +
         name_.capacity() * sizeof(char) + sizeof(model_ids_) +
         model_ids_.capacity() * sizeof(IndexType);
}

std::ostream &operator<<(std::ostream &os, Resource const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", name_ : " << rhs.name_
     << ", model_ids_ : " << rhs.model_ids_ << ")";
  return os;
}

Resource &ResourceMap::tryAddResource(std::string const &name) {
  auto ret = resource_name2id_map_.find(name);
  if (ret == resource_name2id_map_.end()) {  // not exist
    resource_name2id_map_.emplace(name, resources_.size());
    resources_.emplace_back(resources_.size());
    auto &res = resources_.back();
    res.setName(name);
    return res;
  } else {
    return resources_[ret->second];
  }
}

Resource &ResourceMap::tryAddModelResource(ResourceMap::IndexType model_id,
                                           std::string const &resource_name) {
  auto &resource = tryAddResource(resource_name);
  resource.addModel(model_id);
  model2resource_ids_.at(model_id).push_back(resource.id());
  return resource;
}

ResourceMap::IndexType ResourceMap::resourceId(std::string const &name) const {
  auto ret = resource_name2id_map_.find(name);
  if (ret == resource_name2id_map_.end()) {
    return std::numeric_limits<IndexType>::max();
  } else {
    return ret->second;
  }
}

container::ObserverPtr<const Resource> ResourceMap::resource(
    std::string const &name) const {
  auto id = resourceId(name);
  if (id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return resource(id);
  }
}

container::ObserverPtr<Resource> ResourceMap::resource(
    std::string const &name) {
  auto id = resourceId(name);
  if (id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return resource(id);
  }
}

std::vector<ResourceMap::IndexType> const &ResourceMap::modelResourceIds(
    ResourceMap::IndexType model_id) const {
  openparfAssert(model_id < model2resource_ids_.size());
  return model2resource_ids_[model_id];
}

void ResourceMap::copy(ResourceMap const &rhs) {
  this->BaseType::copy(rhs);
  resources_ = rhs.resources_;
  resource_name2id_map_ = rhs.resource_name2id_map_;
  model2resource_ids_ = rhs.model2resource_ids_;
}

void ResourceMap::move(ResourceMap &&rhs) {
  this->BaseType::move(std::move(rhs));
  resources_ = std::move(rhs.resources_);
  resource_name2id_map_ = std::move(rhs.resource_name2id_map_);
  model2resource_ids_ = std::move(rhs.model2resource_ids_);
}

ResourceMap::IndexType ResourceMap::memory() const {
  IndexType ret = this->BaseType::memory() + sizeof(resources_) +
                  sizeof(resource_name2id_map_);
  for (auto const &v : resources_) {
    ret += v.memory();
  }
  ret += (resources_.capacity() - resources_.size()) * sizeof(Resource);
  for (auto const &rv : model2resource_ids_) {
    ret += sizeof(rv) + rv.capacity() * sizeof(IndexType);
  }
  return ret;
}

std::ostream &operator<<(std::ostream &os, ResourceMap const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", resources_ : " << rhs.resources_;
  os << ", resource_name2id_map_ : {";
  const char *delimiter = "";
  for (auto const &kvp : rhs.resource_name2id_map_) {
    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
    delimiter = ", ";
  }
  os << "}";
  os << ", model2resource_ids_ : {";
  delimiter = "";
  for (auto const &rv : rhs.model2resource_ids_) {
    os << delimiter << rv;
    delimiter = ", ";
  }
  os << "}";
  os << ")";
  return os;
}

}  // namespace database

OPENPARF_END_NAMESPACE
