/**
 * @file   inst.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/inst.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Inst::copy(Inst const &rhs) {
  this->BaseType::copy(rhs);
  pin_ids_ = rhs.pin_ids_;
}

void Inst::move(Inst &&rhs) {
  this->BaseType::move(std::move(rhs));
  pin_ids_ = std::move(rhs.pin_ids_);
}

Inst::IndexType Inst::memory() const {
  return this->BaseType::memory() + sizeof(pin_ids_) +
         pin_ids_.capacity() * sizeof(decltype(pin_ids_)::value_type);
}

std::ostream &operator<<(std::ostream &os, Inst const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_;
  os << ", attr_ : " << *rhs.attr_;
  os << ", pin_ids_ : " << rhs.pin_ids_ << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
