/**
 * @file   shape_constr.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/shape_constr.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

std::ostream &operator<<(std::ostream &os,
                         ShapeConstr::ShapeElement const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", offset_ : " << rhs.offset_
     << ", inst_id_ : " << rhs.inst_id_ << ")";
  return os;
}

void ShapeConstr::copy(ShapeConstr const &rhs) {
  this->BaseType::copy(rhs);
  type_ = rhs.type_;
  elements_ = rhs.elements_;
}

void ShapeConstr::move(ShapeConstr &&rhs) {
  this->BaseType::move(std::move(rhs));
  type_ = std::move(rhs.type_);
  elements_ = std::move(rhs.elements_);
}

ShapeConstr::ShapeElement &
ShapeConstr::addShapeElement(ShapeConstr::PointType const &offset,
                             ShapeConstr::IndexType inst_id) {
  elements_.emplace_back();
  auto &element = elements_.back();
  element.setId(numShapeElements() - 1);
  element.setOffset(offset);
  element.setInstId(inst_id);
  return element;
}

ShapeConstr::IndexType ShapeConstr::memory() const {
  return this->BaseType::memory() + sizeof(type_) + sizeof(elements_) +
         elements_.capacity() * sizeof(decltype(elements_)::value_type);
}

std::ostream &operator<<(std::ostream &os, ShapeConstr const &rhs) {
  os << className<decltype(rhs)>() << "("
     << "id_ : " << rhs.id_ << ", type_ : " << rhs.type_;
  os << ", elements_ : " << rhs.elements_ << ")";
  return os;
}

} // namespace database

OPENPARF_END_NAMESPACE
