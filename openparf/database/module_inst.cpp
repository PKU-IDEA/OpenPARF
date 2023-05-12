/**
 * @file   module_inst.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/module_inst.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void ModuleInstNetlist::copy(ModuleInstNetlist const &rhs) {
  netlist_ = rhs.netlist_;
  inst_ids_ = rhs.inst_ids_;
  net_ids_ = rhs.net_ids_;
  pin_ids_ = rhs.pin_ids_;
}

void ModuleInstNetlist::move(ModuleInstNetlist &&rhs) {
  netlist_ = std::move(rhs.netlist_);
  inst_ids_ = std::move(rhs.inst_ids_);
  net_ids_ = std::move(rhs.net_ids_);
  pin_ids_ = std::move(rhs.pin_ids_);
}

ModuleInst::IndexType ModuleInstNetlist::memory() const {
  return sizeof(*this) +
         inst_ids_.capacity() * sizeof(decltype(inst_ids_)::value_type) +
         net_ids_.capacity() * sizeof(decltype(net_ids_)::value_type) +
         pin_ids_.capacity() * sizeof(decltype(pin_ids_)::value_type);
}

std::ostream &operator<<(std::ostream &os, ModuleInstNetlist const &rhs) {
  os << className<decltype(rhs)>() << "(";
  os << "inst_ids_ : " << rhs.instIds() << ", net_ids_ : " << rhs.netIds()
     << ", pin_ids_ : " << rhs.pinIds();
  if (rhs.netlist_) {
    os << ", netlist_ : " << rhs.netlist_->id();
  }
  os << ")";
  return os;
}

void ModuleInst::copy(ModuleInst const &rhs) {
  this->BaseType::copy(rhs);
  netlist_ = rhs.netlist_;
}

void ModuleInst::move(ModuleInst &&rhs) {
  this->BaseType::move(std::move(rhs));
  netlist_ = std::move(rhs.netlist_);
}

ModuleInst::IndexType ModuleInst::memory() const {
  return this->BaseType::memory() + netlist_.memory();
}

std::ostream &operator<<(std::ostream &os, ModuleInst const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", pin_ids_ : " << rhs.pin_ids_;
  os << ", netlist_ : " << rhs.netlist_;
  os << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
