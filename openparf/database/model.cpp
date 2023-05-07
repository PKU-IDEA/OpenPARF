/**
 * @file   model.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "model.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Model::copy(Model const &rhs) {
  this->BaseType::copy(rhs);
  model_type_ = rhs.model_type_;
  size_ = rhs.size_;
  name_ = rhs.name_;
  model_pins_ = rhs.model_pins_;
  model_pin_name2id_map_ = rhs.model_pin_name2id_map_;
  netlist_ = rhs.netlist_;
}

void Model::move(Model &&rhs) {
  this->BaseType::move(std::move(rhs));
  model_type_ = rhs.model_type_;
  size_ = std::move(rhs.size_);
  name_ = std::move(rhs.name_);
  model_pins_ = std::move(rhs.model_pins_);
  model_pin_name2id_map_ = std::move(rhs.model_pin_name2id_map_);
  netlist_ = std::move(rhs.netlist_);
}

ModelPin &Model::tryAddModelPin(std::string const &name) {
  auto ret = modelPinId(name);
  if (ret == std::numeric_limits<IndexType>::max()) {
    model_pin_name2id_map_.emplace(name, model_pins_.size());
    model_pins_.emplace_back(model_pins_.size(), name);
    return model_pins_.back();
  } else {
    return model_pins_[ret];
  }
}

ModelPin::IndexType Model::modelPinId(std::string const &name) const {
  auto ret = model_pin_name2id_map_.find(name);
  if (ret == model_pin_name2id_map_.end()) {
    return std::numeric_limits<IndexType>::max();
  } else {
    return ret->second;
  }
}

container::ObserverPtr<const ModelPin> Model::modelPin(
    std::string const &name) const {
  auto ret = modelPinId(name);
  if (ret == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return model_pins_[ret];
  }
}

container::ObserverPtr<ModelPin> Model::modelPin(std::string const &name) {
  auto ret = modelPinId(name);
  if (ret == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return model_pins_[ret];
  }
}

void Model::setModelPinNet(IndexType m_p, IndexType n) {
   if(!net_to_model_pin_) {
     net_to_model_pin_.emplace();
   }
   if(!model_pin_to_net_) {
     model_pin_to_net_.emplace();
   }
   net_to_model_pin_->emplace(n, m_p);
   model_pin_to_net_->emplace(m_p, n);
}
container::ObserverPtr<Net> Model::getModelPinNet(ModelPin const& mp) {
  if(model_pin_to_net_->find(mp.id()) == model_pin_to_net_->end()) {
      return container::ObserverPtr<Net>();
  } else return netlist_->net(model_pin_to_net_->at(mp.id()));
}
container::ObserverPtr<const Net> Model::getModelPinNet(ModelPin const& mp) const{
  if(model_pin_to_net_->find(mp.id()) == model_pin_to_net_->end()) {
      return container::ObserverPtr<const Net>();
  } else return netlist_->net(model_pin_to_net_->at(mp.id()));
}
container::ObserverPtr<ModelPin> Model::getNetModelPin(Net const& n) {
  if(net_to_model_pin_->find(n.id()) == net_to_model_pin_->end()) {
      return container::ObserverPtr<ModelPin>();
  } else return container::ObserverPtr<ModelPin>(model_pins_.at(net_to_model_pin_->at(n.id())));
}
container::ObserverPtr<const ModelPin> Model::getNetModelPin(Net const& n) const {
  if(net_to_model_pin_->find(n.id()) == net_to_model_pin_->end()) {
      return container::ObserverPtr<const ModelPin>();
  } else return container::ObserverPtr<const ModelPin>(model_pins_.at(net_to_model_pin_->at(n.id())));
}


Inst &Model::tryAddInst(std::string const &name) {
  if (!netlist_) {
    netlist_.emplace();
    netlist_->setId(id());
  }
  if (!inst_name2id_map_) {
    inst_name2id_map_.emplace();
  }
  auto ret = inst_name2id_map_->find(name);
  if (ret == inst_name2id_map_->end()) {
    auto &inst = netlist_->addInst(netlist_->numInsts());
    auto &attr = inst.attr();
    attr.setName(name);
    inst_name2id_map_->emplace(name, inst.id());
    return inst;
  } else {
    return netlist_->inst(ret->second);
  }
}

Model::IndexType Model::instId(std::string const &name) const {
  if (inst_name2id_map_) {
    auto ret = inst_name2id_map_->find(name);
    if (ret != inst_name2id_map_->end()) {
      return ret->second;
    }
  }
  return std::numeric_limits<IndexType>::max();
}

container::ObserverPtr<const Inst> Model::inst(std::string const &name) const {
  auto inst_id = instId(name);
  if (inst_id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return netlist_->inst(inst_id);
  }
}

container::ObserverPtr<Inst> Model::inst(std::string const &name) {
  auto inst_id = instId(name);
  if (inst_id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return netlist_->inst(inst_id);
  }
}

Net &Model::tryAddNet(std::string const &name) {
  if (!netlist_) {
    netlist_.emplace();
    netlist_->setId(id());
  }
  if (!net_name2id_map_) {
    net_name2id_map_.emplace();
  }
  auto ret = net_name2id_map_->find(name);
  if (ret == net_name2id_map_->end()) {
    auto &net = netlist_->addNet(netlist_->numNets());
    auto &attr = net.attr();
    attr.setName(name);
    net_name2id_map_->emplace(name, net.id());
    return net;
  } else {
    return netlist_->net(ret->second);
  }
}

Model::IndexType Model::netId(std::string const &name) const {
  if (net_name2id_map_) {
    auto ret = net_name2id_map_->find(name);
    if (ret != net_name2id_map_->end()) {
      return ret->second;
    }
  }
  return std::numeric_limits<IndexType>::max();
}

container::ObserverPtr<const Net> Model::net(std::string const &name) const {
  auto net_id = netId(name);
  if (net_id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return netlist_->net(net_id);
  }
}

container::ObserverPtr<Net> Model::net(std::string const &name) {
  auto net_id = netId(name);
  if (net_id == std::numeric_limits<IndexType>::max()) {
    return {};
  } else {
    return netlist_->net(net_id);
  }
}

Pin &Model::addPin() {
  if (!netlist_) {
    netlist_.emplace();
    netlist_->setId(id());
  }
  auto &pin = netlist_->addPin(netlist_->numPins());
  auto &attr = pin.attr();
  return pin;
}

Model::IndexType Model::memory() const {
  IndexType ret = this->BaseType::memory() + sizeof(model_type_) +
                  sizeof(model_pins_) + sizeof(name_) + sizeof(size_);
  for (auto const &mpin : model_pins_) {
    ret += mpin.memory();
  }
  ret += (model_pins_.capacity() - model_pins_.size()) *
         sizeof(decltype(model_pins_)::value_type);
  if (netlist_.is_initialized()) {
    ret += netlist_.get().memory();
  }
  return ret;
}

std::ostream &operator<<(std::ostream &os, Model const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", \nmodel_type_ : " << rhs.model_type_
     << ", \nsize_ : " << rhs.size_ << ", \nname_ : " << rhs.name_
     << ", \nmodel_pins_ : " << rhs.model_pins_;
  os << ", \nmodel_pin_name2id_map_ : {";
  const char *delimiter = "";
  for (auto const &kvp : rhs.model_pin_name2id_map_) {
    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
    delimiter = ", ";
  }
  os << "}";
  if (rhs.netlist_.is_initialized()) {
    os << ", \nnetlist_ : " << rhs.netlist_.value();
  }
  os << ")";
  return os;
}

}  // namespace database

OPENPARF_END_NAMESPACE
