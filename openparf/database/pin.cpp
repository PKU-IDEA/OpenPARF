/**
 * @file   pin.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/pin.h"
#include "database/inst.h"
#include "database/net.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void ModelPin::copy(ModelPin const &rhs) {
  this->BaseType::copy(rhs);
  name_ = rhs.name_;
  offset_ = rhs.offset_;
  signal_direct_ = rhs.signal_direct_;
  signal_type_ = rhs.signal_type_;
}

void ModelPin::move(ModelPin &&rhs) {
  this->BaseType::move(std::move(rhs));
  name_ = std::move(rhs.name_);
  offset_ = std::move(rhs.offset_);
  signal_direct_ = std::move(rhs.signal_direct_);
  signal_type_ = std::move(rhs.signal_type_);
}

ModelPin::IndexType ModelPin::memory() const {
  return this->BaseType::memory() + sizeof(offset_) + sizeof(signal_direct_) +
         sizeof(signal_type_);
}

std::ostream &operator<<(std::ostream &os, ModelPin const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", name_ : " << rhs.name_
     << ", offset_ : " << rhs.offset_
     << ", signal_direct_ : " << rhs.signal_direct_
     << ", signal_type_ : " << rhs.signal_type_ << ")";
  return os;
}

void Pin::copy(Pin const &rhs) {
  this->BaseType::copy(rhs);
  model_pin_id_ = rhs.model_pin_id_;
  inst_id_ = rhs.inst_id_;
  net_id_ = rhs.net_id_;
}

void Pin::move(Pin &&rhs) {
  this->BaseType::move(std::move(rhs));
  model_pin_id_ = std::move(rhs.model_pin_id_);
  inst_id_ = std::move(rhs.inst_id_);
  net_id_ = std::move(rhs.net_id_);
}

Pin::IndexType Pin::memory() const {
  return this->BaseType::memory() + sizeof(inst_id_) + sizeof(net_id_);
}

std::ostream &operator<<(std::ostream &os, Pin const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", attr_ : " << *rhs.attr_;
  os << ", model_pin_id_ : " << rhs.model_pin_id_;
  os << ", inst_id_ : " << rhs.inst_id_;
  os << ", net_id_ : " << rhs.net_id_;
  os << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
