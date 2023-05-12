/**
 * @file   Inst.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_INST_H_
#define OPENPARF_DATABASE_INST_H_

#include "database/attr.h"
#include "database/attr_object.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Basic instance or instantiation of a model.
/// It covers instantiation of any type without worrying about its internal
/// implementation. IO is a special case for Cell, where there is an internal
/// pin and external pin, and its model_id_ field is infinity.
class Inst : public AttrObject<InstAttr> {
public:
  using BaseType = AttrObject<InstAttr>;
  using AttrType = BaseType::AttrType;

  /// @brief default constructor
  Inst() : BaseType() {}

  /// @brief constructor
  Inst(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  Inst(Inst const &rhs) { copy(rhs); }

  /// @brief move constructor
  Inst(Inst &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Inst &operator=(Inst const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move constructor
  Inst &operator=(Inst &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief number of pins
  IndexType numPins() const { return pin_ids_.size(); }
  /// @brief getter for a pin
  IndexType pinId(IndexType id) const {
    openparfAssert(id < numPins());
    return pin_ids_[id];
  }

  /// @brief getter for pin
  std::vector<IndexType> const &pinIds() const { return pin_ids_; }

  /// @brief getter for pin
  std::vector<IndexType> &pinIds() { return pin_ids_; }

  /// @brief add a pin
  void addPin(IndexType p) { pin_ids_.push_back(p); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(Inst const &rhs);
  /// @brief move object
  void move(Inst &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Inst const &rhs);

  std::vector<IndexType> pin_ids_; ///< pin connection of the instance
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
