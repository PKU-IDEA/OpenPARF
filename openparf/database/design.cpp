/**
 * @file   design.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/design.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>


OPENPARF_BEGIN_NAMESPACE

namespace database {

Model &Design::addModel(std::string const &name) {
  auto ret = model_name2id_map_.emplace(name, models_.size());
  if (ret.second) {
    models_.emplace_back(models_.size());
    auto &model = models_.back();
    model.setName(name);
    return model;
  } else {
    openparfPrint(kError, "model %s already defined, ignored\n", name.c_str());
    return models_[ret.first->second];
  }
}

Design::IndexType Design::modelId(std::string const &name) const {
  auto ret = model_name2id_map_.find(name);
  if (ret == model_name2id_map_.end()) {
    return std::numeric_limits<IndexType>::max();
  } else {
    return ret->second;
  }
}

container::ObserverPtr<const ModuleInst> Design::moduleInst(Design::IndexType i) const {
  if (i < numModuleInsts()) {
    return (*module_insts_)[i];
  } else {
    return {};
  }
}

container::ObserverPtr<ModuleInst> Design::moduleInst(Design::IndexType i) {
  if (i < numModuleInsts()) {
    return (*module_insts_)[i];
  } else {
    return {};
  }
}

container::ObserverPtr<const ModuleInst> Design::topModuleInst() const {
  return moduleInst(top_module_inst_);
}

container::ObserverPtr<ModuleInst> Design::topModuleInst() {
  return moduleInst(top_module_inst_);
}

ShapeConstr &Design::addShapeConstr(ShapeConstrType const &t) {
  shape_constrs_.emplace_back(shape_constrs_.size());
  auto &constr = shape_constrs_.back();
  constr.setType(t);
  return constr;
}

void Design::copy(Design const &rhs) {
  this->BaseType::copy(rhs);

  // copy models
  models_  = rhs.models_;

  netlist_ = rhs.netlist_;

  // copy hierarchy
  if (rhs.module_insts_) {
    if (!module_insts_) {
      module_insts_.emplace();
    }
    module_insts_->clear();
    module_insts_->reserve(rhs.module_insts_->size());
    for (IndexType i = 0, ie = rhs.module_insts_->size(); i != ie; ++i) {
      auto &src = (*rhs.module_insts_)[i];
      module_insts_->emplace_back(*netlist_);
      auto &tgt = module_insts_->back();
      tgt.setId(src.id());
      tgt.pinIds()                   = src.pinIds();

      ModuleInstNetlist &module_impl = tgt.netlist();
      for (auto inst_id : src.netlist().instIds()) {
        module_impl.addInst(inst_id);
      }
      for (auto net_id : src.netlist().netIds()) {
        module_impl.addNet(net_id);
      }
      for (auto pin_id : src.netlist().pinIds()) {
        module_impl.addPin(pin_id);
      }
    }
  }

  // set top module instance
  top_module_inst_ = rhs.top_module_inst_;

  // shape constraints
  shape_constrs_   = rhs.shape_constrs_;
}

void Design::move(Design &&rhs) {
  this->BaseType::move(std::move(rhs));

  models_          = std::move(rhs.models_);
  module_insts_    = std::move(rhs.module_insts_);
  netlist_         = std::move(rhs.netlist_);
  top_module_inst_ = std::exchange(rhs.top_module_inst_, std::numeric_limits<IndexType>::max());
  shape_constrs_   = std::move(shape_constrs_);
}

Design::IndexType Design::memory() const {
  IndexType ret = this->BaseType::memory() + sizeof(models_) + sizeof(model_name2id_map_) + sizeof(module_insts_) +
                  sizeof(top_module_inst_) + sizeof(shape_constrs_);

  for (auto const &v : models_) {
    ret += v.memory();
  }

  for (auto kvp : model_name2id_map_) {
    ret += kvp.first.length() * sizeof(char) + sizeof(kvp.second);
  }

  ret += sizeof(netlist_) + netlist_->memory();

  if (module_insts_) {
    for (auto const &v : *module_insts_) {
      ret += v.memory();
    }
  }

  for (auto const &v : shape_constrs_) {
    ret += v.memory();
  }

  return ret;
}

std::ostream &operator<<(std::ostream &os, Design const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", \nmodels : [" << rhs.models_.size() << "][";
  const char *delimiter = "";
  for (auto const &v : rhs.models_) {
    os << delimiter << v;
    delimiter = ", \n";
  }
  os << " ]";
  os << ", \nmodel_name2id_map_ : {";
  delimiter = "";
  for (auto kvp : rhs.model_name2id_map_) {
    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
    delimiter = ", ";
  }
  os << "}";
  if (rhs.module_insts_) {
    os << ", \nnetlist_ : " << *rhs.netlist_;
    os << ", \nmodule_insts_ : [" << rhs.module_insts_->size() << "][";
    delimiter = "";
    for (auto const &v : *rhs.module_insts_) {
      os << delimiter << v;
      delimiter = ", \n";
    }
    os << "]";
  }
  os << ", \nshape_constrs_ : [" << rhs.shape_constrs_.size() << "][";
  delimiter = "";
  for (auto const &v : rhs.shape_constrs_) {
    os << delimiter << v;
    delimiter = ", \n";
  }
  os << "]";
  os << ")";
  return os;
}

Design::IndexType Design::uniquify(Design::IndexType model_id) {
  auto const &model = models_.at(model_id);
  if (model.modelType() == ModelType::kModule) {
    // clear current instantiation
    netlist_.reset();
    module_insts_.reset();

    // recursive instantiation
    auto sub_module_inst = tryAddModuleInst(model);
    this->uniquifyKernel(sub_module_inst);
    top_module_inst_ = sub_module_inst->id();

    return top_module_inst_;
  }
  return top_module_inst_;
}

void Design::uniquifyKernel(ModuleInst *module_inst) {
  auto const &model = models_.at(module_inst->attr().modelId());
  openparfAssert(model.modelType() == ModelType::kModule);
  openparfAssertMsg(model.netlist(), "module %s, model %s has no netlist", module_inst->attr().name().c_str(),
          model.name().c_str());

  // mark whether an inst or a net has already been created
  std::vector<IndexType> model_inst2inst(model.netlist()->numInsts(), std::numeric_limits<IndexType>::max());
  std::vector<IndexType> model_net2net(model.netlist()->numNets() + 1, std::numeric_limits<IndexType>::max());

  // create instances
  for (auto const &model_inst : model.netlist()->insts()) {
    auto &inst                       = netlist_->addInst();
    model_inst2inst[model_inst.id()] = inst.id();

    inst.attr()                      = model_inst.attr();
    if (module_inst->attr().name().empty()) {
      inst.attr().setName(model_inst.attr().name());
    } else {
      inst.attr().setName(module_inst->attr().name() + "/" + model_inst.attr().name());
    }

    module_inst->netlist().addInst(inst.id());
  }

  // create nets
  for (auto const &model_net : model.netlist()->nets()) {
    auto &net                     = netlist_->addNet();
    model_net2net[model_net.id()] = net.id();

    net.attr()                    = model_net.attr();
    if (module_inst->attr().name().empty()) {
      net.attr().setName(model_net.attr().name());
    } else {
      net.attr().setName(module_inst->attr().name() + "/" + model_net.attr().name());
    }

    module_inst->netlist().addNet(net.id());
  }

  // create VDD/VSS net
  auto &vddvss_net     = netlist_->addNet();
  vddvss_net_id_       = vddvss_net.id();
  model_net2net.back() = vddvss_net.id();
  if (module_inst->attr().name().empty()) {
    vddvss_net.attr().setName("OPENPARF_VDDVSS");
  } else {
    vddvss_net.attr().setName(module_inst->attr().name() + "/" + "OPENPARF_VDDVSS");
  }
  module_inst->netlist().addNet(vddvss_net.id());

  // pins
  for (auto const &model_pin : model.netlist()->pins()) {
    auto const &model_inst = model.netlist()->inst(model_pin.instId());
    auto const &model_net  = model.netlist()->net(model_pin.netId());

    auto       &pin        = netlist_->addPin();
    auto       &inst       = netlist_->inst(model_inst2inst[model_inst.id()]);
    auto       &net        = netlist_->net(model_net2net[model_net.id()]);

    pin.setModelPinId(model_pin.modelPinId());
    pin.setInstId(inst.id());
    pin.setNetId(net.id());
    pin.attr() = model_pin.attr();

    inst.addPin(pin.id());
    net.addPin(pin.id());

    module_inst->netlist().addPin(pin.id());
  }

  // check floating control signals
  // FIXME(Jing Mai): to align with the ISPD2017 bookshelf benchmark, we only check floating control singals for FFs.

  std::vector<uint8_t> mpin_markers;
  IndexType            num_float_pins[2] = {0, 0};
  for (auto const &model_inst : model.netlist()->insts()) {
    auto const &sub_module_model = models_.at(model_inst.attr().modelId());
    auto        inst_id          = model_inst2inst[model_inst.id()];
    auto const &inst             = module_inst->netlist().inst(inst_id);
    // only happens when an instance has fewer pins than its model
    if (sub_module_model.modelType() == ModelType::kCell && inst.numPins() < sub_module_model.numModelPins()) {
      mpin_markers.assign(sub_module_model.numModelPins(), 0);
      // traverse all pins of the instance
      for (auto const &pin_id : inst.pinIds()) {
        auto const &pin  = module_inst->netlist().pin(pin_id);
        auto const &mpin = sub_module_model.modelPin(pin.modelPinId());
        if (mpin.signalType() == SignalType::kClock || mpin.signalType() == SignalType::kControlSR ||
                mpin.signalType() == SignalType::kControlCE) {
          // cover model pin
          mpin_markers[mpin.id()] = 1;
        }
      }

      // traverse all pins of the model
      for (IndexType i = 0; i < sub_module_model.numModelPins(); ++i) {
        auto const &mpin = sub_module_model.modelPin(i);
        openparfAssert(mpin.id() == i);
        // clock cannot be floating
        if (mpin.signalType() == SignalType::kClock) {
          if (mpin_markers[i] == 0) {   // not covered
            // TODO(Jing Mai): we ignore the floating clocks for the time being as the modeling for xarch benchmarks is
            // not complete.
            openparfPrint(kWarn, "Inst %s (%u) of model %s has floating clock signal %s\n", inst.attr().name().c_str(),
                    inst.id(), model.name().c_str(), mpin.name().c_str());
          } else {   // covered
            mpin_markers[i] = 0;
          }
        }
        // ControlSR and ControlCE can be floating
        // this will be regarded as VDD/VSS net
        if (mpin.signalType() == SignalType::kControlSR || mpin.signalType() == SignalType::kControlCE) {
          if (mpin_markers[i] == 0) {   // not covered
            if (mpin.signalType() == SignalType::kControlSR) {
              num_float_pins[0] += 1;
            } else {
              num_float_pins[1] += 1;
            }
            // add to VDD/VSS net
            auto &float_pin  = netlist_->addPin();
            auto &float_inst = netlist_->inst(model_inst2inst[model_inst.id()]);
            auto &float_net  = netlist_->net(vddvss_net.id());
            openparfAssert(float_inst.id() == inst.id());
            openparfAssert(float_net.id() == vddvss_net.id());

            float_pin.setModelPinId(mpin.id());
            float_pin.setInstId(float_inst.id());
            float_pin.setNetId(float_net.id());
            float_pin.attr().setSignalDirect(mpin.signalDirect());

            float_inst.addPin(float_pin.id());
            float_net.addPin(float_pin.id());

            module_inst->netlist().addPin(float_pin.id());
          } else {   // covered
            mpin_markers[i] = 0;
          }
        }
      }
    }
  }
  openparfPrint(kInfo, "detect %u ControlSR and %u ControlCE floating pins, add to VDD/VSS net %s (%u)\n",
          num_float_pins[0], num_float_pins[1], vddvss_net.attr().name().c_str(), vddvss_net.id());

  // recursion
  for (auto const &model_inst : model.netlist()->insts()) {
    auto const &sub_module_model = models_.at(model_inst.attr().modelId());
    if (sub_module_model.modelType() == ModelType::kModule) {
      auto sub_module_inst = tryAddModuleInst(sub_module_model);
      this->uniquifyKernel(sub_module_inst);
    }
  }
}

ModuleInst *Design::tryAddModuleInst(Model const &model) {
  if (model.modelType() == ModelType::kModule) {
    if (!module_insts_) {
      module_insts_.emplace();
      netlist_.emplace(0);
    }
    module_insts_->emplace_back(module_insts_->size(), *netlist_);
    auto &module_inst      = module_insts_->back();
    auto &module_inst_attr = module_inst.attr();
    module_inst_attr.setModelId(model.id());
    return &module_inst;
  } else {
    return nullptr;
  }
}

}   // namespace database

OPENPARF_END_NAMESPACE
