/**
 * @file   super_logic_region.cpp
 * @author Runzhe Tao
 * @date   Nov 2023
 */

#include "database/super_logic_region.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void SuperLogicRegion::copy(SuperLogicRegion const &rhs) {
  this->BaseType::copy(rhs);
  name_   = rhs.name_;
  type_   = rhs.type_;
  bbox_   = rhs.bbox_;
  idx_    = rhs.idx_;
  idy_    = rhs.idy_;
  width_  = rhs.width_;
  height_ = rhs.height_;
}

void SuperLogicRegion::move(SuperLogicRegion &&rhs) {
  this->BaseType::move(std::move(rhs));
  name_   = std::move(rhs.name_);
  type_   = std::move(rhs.type_);
  bbox_   = std::move(rhs.bbox_);
  idx_    = std::move(rhs.idx_);
  idy_    = std::move(rhs.idy_);
  width_  = std::move(rhs.width_);
  height_ = std::move(rhs.height_);
}

}  // namespace database

OPENPARF_END_NAMESPACE