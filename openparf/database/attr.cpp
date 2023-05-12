/**
 * @file   attr.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */
#include "database/attr.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void BaseAttr::copy(BaseAttr const &rhs) { this->BaseType::copy(rhs); }

void BaseAttr::move(BaseAttr &&rhs) { this->BaseType::move(std::move(rhs)); }

/// @brief overload output stream
std::ostream &operator<<(std::ostream &os, BaseAttr const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ")";
  return os;
}

void InstAttr::copy(InstAttr const &rhs) {
  this->BaseType::copy(rhs);
  model_id_ = rhs.model_id_;
  place_status_ = rhs.place_status_;
  loc_ = rhs.loc_;
  name_ = rhs.name_;
}

void InstAttr::move(InstAttr &&rhs) {
  this->BaseType::move(std::move(rhs));
  model_id_ = std::move(rhs.model_id_);
  place_status_ = std::move(place_status_);
  loc_ = std::move(rhs.loc_);
  name_ = std::move(rhs.name_);
}

InstAttr::IndexType InstAttr::memory() const {
  return this->BaseType::memory() + sizeof(model_id_) + sizeof(place_status_) +
         sizeof(loc_) + sizeof(name_) +
         name_.capacity() * sizeof(decltype(name_)::value_type);
}

std::ostream &operator<<(std::ostream &os, InstAttr const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", model_id_ : " << rhs.model_id_
     << ", place_status_ : " << rhs.place_status_ << ", loc_ : " << rhs.loc_
     << ", name_ : " << rhs.name_ << ")";
  return os;
}

void NetAttr::copy(NetAttr const &rhs) {
  this->BaseType::copy(rhs);
  weight_ = rhs.weight_;
  name_ = rhs.name_;
}

void NetAttr::move(NetAttr &&rhs) {
  this->BaseType::move(std::move(rhs));
  weight_ = rhs.weight_;
  name_ = std::move(rhs.name_);
}

NetAttr::IndexType NetAttr::memory() const {
  return this->BaseType::memory() + sizeof(weight_) + sizeof(name_) +
         name_.capacity() * sizeof(decltype(name_)::value_type);
}

std::ostream &operator<<(std::ostream &os, NetAttr const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", weight_ : " << rhs.weight_
     << ", name_ : " << rhs.name_ << ")";
  return os;
}

void PinAttr::copy(PinAttr const &rhs) {
  this->BaseType::copy(rhs);
  signal_direct_ = rhs.signal_direct_;
}

void PinAttr::move(PinAttr &&rhs) {
  this->BaseType::move(std::move(rhs));
  signal_direct_ = std::exchange(rhs.signal_direct_, SignalDirection::kUnknown);
}

PinAttr::IndexType PinAttr::memory() const {
  return this->BaseType::memory() + sizeof(signal_direct_);
}

std::ostream &operator<<(std::ostream &os, PinAttr const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", signal_direct_ : " << rhs.signal_direct_
     << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
