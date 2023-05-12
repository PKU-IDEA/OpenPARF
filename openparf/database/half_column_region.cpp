/**
 * @file   half_column_region.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/half_column_region.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void HalfColumnRegion::copy(HalfColumnRegion const &rhs) {
  this->BaseType::copy(rhs);
  clock_region_id_ = rhs.clock_region_id_;
  bbox_ = rhs.bbox_;
}

void HalfColumnRegion::move(HalfColumnRegion &&rhs) {
    this->BaseType::move(std::move(rhs));
    clock_region_id_ = std::move(rhs.clock_region_id_);
    bbox_            = std::move(rhs.bbox_);
}

HalfColumnRegion::IndexType HalfColumnRegion::memory() const {
  return this->BaseType::memory() + sizeof(clock_region_id_) + bbox_.memory();
}

std::ostream &operator<<(std::ostream &os, HalfColumnRegion const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", clock_region_id_ : " << rhs.clock_region_id_
     << ", bbox_ : " << rhs.bbox_ << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
