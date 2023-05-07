/**
 * @file   net.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */
#include "database/net.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Net::copy(Net const &rhs) {
  this->BaseType::copy(rhs);
  pin_ids_ = rhs.pin_ids_;
}

void Net::move(Net &&rhs) {
  this->BaseType::move(std::move(rhs));
  pin_ids_ = std::move(rhs.pin_ids_);
}

Net::IndexType Net::memory() const {
  return this->BaseType::memory() + sizeof(pin_ids_) +
         pin_ids_.capacity() * sizeof(decltype(pin_ids_)::value_type);
}

std::ostream &operator<<(std::ostream &os, Net const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", attr_ : " << *rhs.attr_
     << ", pin_ids_ : " << rhs.pin_ids_ << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
