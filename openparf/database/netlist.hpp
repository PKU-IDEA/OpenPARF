/**
 * @file   netlist.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_NETLIST_HPP_
#define OPENPARF_DATABASE_NETLIST_HPP_

#include "database/object.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief A general representation for flat circuit graph.
/// Note that hierarchy is not handled.
/// Here "flat" means we will not go into InstType for further expansion.
///
/// To represent a circuit graph, three components are required:
/// instances, nets, and pins.
/// An instance contains multiple pins, so as a net.
/// A pin records its parent instance and net.
template <typename InstType, typename NetType, typename PinType>
class Netlist : public Object {
public:
  using BaseType = Object;

  /// @brief default constructor
  Netlist() : BaseType() {}

  /// @brief constructor
  Netlist(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  Netlist(Netlist const &rhs) { copy(rhs); }

  /// @brief move constructor
  Netlist(Netlist &&rhs) { move(std::move(rhs)); }

  /// @brief copy assignment
  Netlist &operator=(Netlist const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Netlist &operator=(Netlist &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief number of instance
  IndexType numInsts() const { return insts_.size(); }

  /// @brief getter for instance
  InstType const &inst(IndexType id) const {
    openparfAssert(id < numInsts());
    return insts_[id];
  }

  /// @brief getter for instance
  InstType &inst(IndexType id) {
    openparfAssert(id < numInsts());
    return insts_[id];
  }

  /// @brief getter for instance
  std::vector<InstType> const &insts() const { return insts_; }

  /// @brief getter for instance
  std::vector<InstType> &insts() { return insts_; }

  /// @brief add an instance
  template <class... Args> InstType &addInst(Args... args) {
    insts_.emplace_back(args...);
    insts_.back().setId(insts_.size() - 1);
    return insts_.back();
  }

  /// @brief number of net
  IndexType numNets() const { return nets_.size(); }

  /// @brief getter for net
  NetType const &net(IndexType id) const {
    openparfAssert(id < numNets());
    return nets_[id];
  }

  /// @brief getter for net
  NetType &net(IndexType id) {
    openparfAssert(id < numNets());
    return nets_[id];
  }

  /// @brief getter for net
  std::vector<NetType> const &nets() const { return nets_; }

  /// @brief getter for net
  std::vector<NetType> &nets() { return nets_; }

  /// @brief add an net
  template <class... Args> NetType &addNet(Args... args) {
    nets_.emplace_back(args...);
    nets_.back().setId(nets_.size() - 1);
    return nets_.back();
  }

  /// @brief number of pin
  IndexType numPins() const { return pins_.size(); }

  /// @brief getter for pin
  PinType const &pin(IndexType id) const {
    openparfAssert(id < numPins());
    return pins_[id];
  }

  /// @brief getter for pin
  PinType &pin(IndexType id) {
    openparfAssert(id < numPins());
    return pins_[id];
  }

  /// @brief getter for pin
  std::vector<PinType> const &pins() const { return pins_; }

  /// @brief getter for pin
  std::vector<PinType> &pins() { return pins_; }

  /// @brief add an pin
  template <class... Args> PinType &addPin(Args... args) {
    pins_.emplace_back(args...);
    pins_.back().setId(pins_.size() - 1);
    return pins_.back();
  }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(Netlist const &rhs);
  /// @brief move object
  void move(Netlist &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Netlist const &rhs) {
    os << className<decltype(rhs)>() << "("
       << "id_ : " << rhs.id_ << ", \ninsts_ : " << rhs.insts_
       << ", \nnets_ : " << rhs.nets_ << ", \npins_ : " << rhs.pins_;
    os << ")";
    return os;
  }

  /// netlist
  std::vector<InstType> insts_; ///< instances
  std::vector<NetType> nets_;   ///< nets
  std::vector<PinType> pins_;   ///< pins
};

template <typename InstType, typename NetType, typename PinType>
typename Netlist<InstType, NetType, PinType>::IndexType
Netlist<InstType, NetType, PinType>::memory() const {
  IndexType size = sizeof(insts_) + sizeof(nets_) + sizeof(pins_);
  for (auto const &v : insts_) {
    size += v.memory();
  }
  size += (insts_.capacity() - insts_.size()) * sizeof(InstType);
  for (auto const &v : nets_) {
    size += v.memory();
  }
  size += (nets_.capacity() - nets_.size()) * sizeof(NetType);
  for (auto const &v : pins_) {
    size += v.memory();
  }
  size += (pins_.capacity() - pins_.size()) * sizeof(PinType);

  return size;
}

template <typename InstType, typename NetType, typename PinType>
void Netlist<InstType, NetType, PinType>::copy(
    Netlist<InstType, NetType, PinType> const &rhs) {
  this->BaseType::copy(rhs);

  insts_ = rhs.insts_;
  nets_ = rhs.nets_;
  pins_ = rhs.pins_;
}

template <typename InstType, typename NetType, typename PinType>
void Netlist<InstType, NetType, PinType>::move(
    Netlist<InstType, NetType, PinType> &&rhs) {
  this->BaseType::move(std::move(rhs));
  insts_ = std::move(rhs.insts_);
  nets_ = std::move(rhs.nets_);
  pins_ = std::move(rhs.pins_);
}

} // namespace database

OPENPARF_END_NAMESPACE

#endif
