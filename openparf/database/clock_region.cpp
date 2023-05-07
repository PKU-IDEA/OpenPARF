/**
 * @file   clock_region.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/clock_region.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void ClockRegion::copy(ClockRegion const &rhs) {
    this->BaseType::copy(rhs);
    name_                   = rhs.name_;
    bbox_                   = rhs.bbox_;
    hc_xmin_                = std::move(rhs.hc_xmin_);
    hc_xmax_                = std::move(rhs.hc_xmax_);
    half_column_region_ids_ = rhs.half_column_region_ids_;
    num_sites_              = rhs.num_sites_;
}

void ClockRegion::move(ClockRegion &&rhs) {
    this->BaseType::move(std::move(rhs));
    name_                   = std::move(rhs.name_);
    bbox_                   = std::move(rhs.bbox_);
    hc_xmin_                = std::move(rhs.hc_xmin_);
    hc_xmax_                = std::move(rhs.hc_xmax_);
    half_column_region_ids_ = std::move(rhs.half_column_region_ids_);
    num_sites_              = std::move(rhs.num_sites_);
}

ClockRegion::IndexType ClockRegion::memory() const {
    return this->BaseType::memory() + sizeof(name_) + name_.capacity() * sizeof(char) +
           bbox_.memory() + +sizeof(hc_xmin_) + sizeof(hc_xmax_) + sizeof(half_column_region_ids_) +
           sizeof(num_sites_) +
           half_column_region_ids_.capacity() *
                   sizeof(decltype(half_column_region_ids_)::value_type) +
           num_sites_.capacity() * sizeof(decltype(num_sites_)::value_type);
}

std::ostream &operator<<(std::ostream &os, ClockRegion const &rhs) {
    os << className<decltype(rhs)>() << "("
       << "id_ : " << rhs.id_ << ", name_ : " << rhs.name_ << ", bbox_ : " << rhs.bbox_
       << ", hc_xmin : " << rhs.hc_xmin_ << ", hc_xmax : " << rhs.hc_xmax_
       << ", half_column_region_ids_ : " << rhs.half_column_region_ids_
       << ", num_sites_ : " << rhs.num_sites_;
    os << ")";
    return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
