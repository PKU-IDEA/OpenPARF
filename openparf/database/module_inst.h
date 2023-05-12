/**
 * @file   module_inst.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_MODULE_INST_H_
#define OPENPARF_DATABASE_MODULE_INST_H_

#include "container/container.hpp"
// local dependency
#include "database/inst.h"
#include "database/net.h"
#include "database/netlist.hpp"
#include "database/pin.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Internal implementation of the module instance;
/// Only pointers of the netlist instances are recorded.
/// That is why an individual class is necessary.
class ModuleInstNetlist {
 public:
  using IndexType = Object::IndexType;
  using NetlistType = Netlist<Inst, Net, Pin>;

  /// @brief constructor
  ModuleInstNetlist(NetlistType &netlist) : netlist_(netlist) {}

  /// @brief copy constructor
  ModuleInstNetlist(ModuleInstNetlist const &rhs) { copy(rhs); }

  /// @brief move constructor
  ModuleInstNetlist(ModuleInstNetlist &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  ModuleInstNetlist &operator=(ModuleInstNetlist const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  ModuleInstNetlist &operator=(ModuleInstNetlist &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief number of instance
  IndexType numInsts() const { return inst_ids_.size(); }

  /// @brief getter for instance
  IndexType instId(IndexType id) const {
    openparfAssert(netlist_);
    openparfAssert(id < numInsts());
    return inst_ids_[id];
  }

  /// @brief getter for instance
  Inst const &inst(IndexType inst_id) const { return netlist_->inst(inst_id); }

  /// @brief getter for instance
  Inst &inst(IndexType inst_id) { return netlist_->inst(inst_id); }

  /// @brief getter for instance
  std::vector<IndexType> const &instIds() const { return inst_ids_; }

  /// @brief add an instance
  Inst &addInst(IndexType id) {
    openparfAssert(netlist_);
    openparfAssert(id < netlist_->numInsts());
    inst_ids_.emplace_back(id);
    return netlist_->inst(id);
  }

  /// @brief number of net
  IndexType numNets() const { return net_ids_.size(); }

  /// @brief getter for net
  IndexType netId(IndexType id) const {
    openparfAssert(id < numNets());
    return net_ids_[id];
  }

  /// @brief getter for net
  Net const &net(IndexType net_id) const { return netlist_->net(net_id); }

  /// @brief getter for net
  Net &net(IndexType net_id) { return netlist_->net(net_id); }

  /// @brief getter for net
  std::vector<IndexType> const &netIds() const { return net_ids_; }

  /// @brief add an net
  Net &addNet(IndexType id) {
    openparfAssert(netlist_);
    openparfAssert(id < netlist_->numNets());
    net_ids_.emplace_back(id);
    return netlist_->net(id);
  }

  /// @brief number of pins
  IndexType numPins() const { return pin_ids_.size(); }

  /// @brief getter for pin
  IndexType pinId(IndexType id) const {
    openparfAssert(id < numPins());
    return pin_ids_[id];
  }

  /// @brief getter for pin
  Pin const &pin(IndexType pin_id) const { return netlist_->pin(pin_id); }

  /// @brief getter for pin
  Pin &pin(IndexType pin_id) { return netlist_->pin(pin_id); }

  /// @brief getter for pin
  std::vector<IndexType> const &pinIds() const { return pin_ids_; }

  /// @brief add an pin
  Pin &addPin(IndexType id) {
    openparfAssert(netlist_);
    openparfAssert(id < netlist_->numPins());
    pin_ids_.emplace_back(id);
    return netlist_->pin(id);
  }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(ModuleInstNetlist const &rhs);
  /// @brief move object
  void move(ModuleInstNetlist &&rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  ModuleInstNetlist const &rhs);

  container::ObserverPtr<NetlistType>
      netlist_;                      ///< reference to the parent netlist
  std::vector<IndexType> inst_ids_;  ///< index of instances in the module inst
  std::vector<IndexType> net_ids_;   ///< index of nets in the module inst
  std::vector<IndexType> pin_ids_;   ///< index of pins in the module inst
};

/// @brief An instance or instantiation of a module. We care about
/// its internal implementation.
class ModuleInst : public Inst {
 public:
  using BaseType = Inst;
  using NetlistType = ModuleInstNetlist::NetlistType;

  /// @brief constructor
  ModuleInst(NetlistType &n) : BaseType(), netlist_(n) {}

  /// @brief constructor
  ModuleInst(IndexType id, NetlistType &n) : BaseType(id), netlist_(n) {}

  /// @brief copy constructor
  ModuleInst(ModuleInst const &rhs) : BaseType(rhs), netlist_(rhs.netlist_) {}

  /// @brief move constructor
  ModuleInst(ModuleInst &&rhs) noexcept
      : BaseType(std::move(rhs)), netlist_(std::move(rhs.netlist_)) {}

  /// @brief copy constructor
  ModuleInst &operator=(ModuleInst const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move constructor
  ModuleInst &operator=(ModuleInst &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for implementation
  ModuleInstNetlist const &netlist() const { return netlist_; }

  /// @brief getter for implementation
  ModuleInstNetlist &netlist() { return netlist_; }

  /// @brief setter for implementation
  void setNetlist(ModuleInstNetlist const &v) { netlist_ = v; }

  /// @brief setter for implementation
  void setNetlist(ModuleInstNetlist &&v) { netlist_ = std::move(v); }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(ModuleInst const &rhs);
  /// @brief move object
  void move(ModuleInst &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, ModuleInst const &rhs);

  ModuleInstNetlist
      netlist_;  ///< internal instantiation circuit for the module
};

}  // namespace database

OPENPARF_END_NAMESPACE

#endif
