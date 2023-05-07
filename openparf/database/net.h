/**
 * @file   net.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_NET_H_
#define OPENPARF_DATABASE_NET_H_

#include "database/attr.h"
#include "database/attr_object.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Net for an interconnection,
/// like a hyperedge in a hypergraph
class Net : public AttrObject<NetAttr> {
public:
  using BaseType = AttrObject<NetAttr>;
  using AttrType = BaseType::AttrType;

  /// @brief default constructor
  Net() : BaseType() {}

  /// @brief constructor
  Net(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  Net(Net const &rhs) { copy(rhs); }

  /// @brief move constructor
  Net(Net &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Net &operator=(Net const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move constructor
  Net &operator=(Net &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief number of pins
  IndexType numPins() const { return pin_ids_.size(); }
  /// @brief getter for a pin index
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
  void copy(Net const &rhs);
  /// @brief move object
  void move(Net &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Net const &rhs);

  std::vector<IndexType> pin_ids_; ///< pin connection in the net
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
