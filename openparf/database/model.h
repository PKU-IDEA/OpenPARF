/**
 * @file   model.h
 * @author Yibo Lin
 * @date   Mar 2020
 */
#ifndef OPENPARF_DATABASE_MODEL_H_
#define OPENPARF_DATABASE_MODEL_H_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

// 3rd party libraries
#include "boost/optional.hpp"

// project headers
#include "container/container.hpp"
#include "database/attr.h"
#include "database/inst.h"
#include "database/net.h"
#include "database/netlist.hpp"
#include "database/pin.h"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Describe all referenceable models,
/// such as Standard Cell, IOPin, Module.
/// IOPin is a bit special, there will be only one.
class Model : public Object {
 public:
  using BaseType    = Object;
  using SizeType    = geometry::Size<CoordinateType, 2>;
  /// @brief internal circuit for the module
  /// Inst is the basic instance without worrying about internal
  /// implementations. It is created for hierarchical Module only.
  using NetlistType = Netlist<Inst, Net, Pin>;

  /// @brief default constructor
  Model() : BaseType() { model_type_ = ModelType::kUnknown; }

  /// @brief constructor
  explicit Model(IndexType id) : BaseType(id) { model_type_ = ModelType::kUnknown; }

  /// @brief copy constructor
  Model(Model const &rhs) { copy(rhs); }

  /// @brief move constructor
  Model(Model &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Model &operator=(Model const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Model &operator=(Model &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for model type
  ModelType       modelType() const { return model_type_; }

  /// @brief setter for model type
  void            setModelType(ModelType const &v) { model_type_ = v; }

  /// @brief number of pins
  IndexType       numModelPins() const { return model_pins_.size(); }
  /// @brief getter for a pin
  ModelPin const &modelPin(IndexType id) const {
    openparfAssert(id < numModelPins());
    return model_pins_[id];
  }
  /// @brief getter for a pin
  ModelPin &modelPin(IndexType id) {
    openparfAssert(id < numModelPins());
    return model_pins_[id];
  }
  /// @brief add a model pin if not exist; otherwise, do not add.
  ModelPin                              &tryAddModelPin(std::string const &name);

  /// @brief get the index of a model pin, given model pin name; infinity if not
  /// found
  IndexType                              modelPinId(std::string const &name) const;

  /// @brief get a model pin, given model pin name; nullptr if not found
  container::ObserverPtr<const ModelPin> modelPin(std::string const &name) const;

  /// @brief get a model pin, given model pin name; nullptr if not found
  container::ObserverPtr<ModelPin>       modelPin(std::string const &name);

  /// @brief Setup and retrive mapping between the model's io pins and nets.
  /// Note that instances and submodules in the netlist CANNOT directly connect to
  /// the module's io pins. They connect to a net, which in turn connect to a Module IO pin.
  void                                   setModelPinNet(IndexType modelPinId, IndexType netId);
  container::ObserverPtr<Net>            getModelPinNet(ModelPin const &);
  container::ObserverPtr<const Net>      getModelPinNet(ModelPin const &) const;
  container::ObserverPtr<ModelPin>       getNetModelPin(Net const &);
  container::ObserverPtr<const ModelPin> getNetModelPin(Net const &) const;

  /// @brief getter for name
  std::string const                     &name() const { return name_; }

  /// @brief setter for name
  void                                   setName(std::string const &name) { name_ = name; }

  /// @brief getter for size
  SizeType const                        &size() const { return size_; }

  /// @brief setter for size
  void                                   setSize(SizeType const &v) { size_ = v; }

  /// @brief getter for implementation
  boost::optional<NetlistType> const    &netlist() const { return netlist_; }

  /// @brief getter for implementation
  boost::optional<NetlistType>          &netlist() { return netlist_; }

  /// @brief setter for implementation
  void                                   setNetlist(NetlistType const &v) { netlist_ = v; }
  /// @brief setter for implementation
  void                                   setNetlist(NetlistType &&v) { netlist_ = std::move(v); }

  /// @brief shortcut getter for instance
  container::ObserverPtr<const Inst>     inst(IndexType i) const {
        if (netlist_) {
          return netlist_->inst(i);
    }
        return {};
  }

  /// @brief shortcut getter for instance
  container::ObserverPtr<Inst> inst(IndexType i) {
    if (netlist_) {
      return netlist_->inst(i);
    }
    return {};
  }

  /// @brief shortcut getter for net
  container::ObserverPtr<const Net> net(IndexType i) const {
    if (netlist_) {
      return netlist_->net(i);
    }
    return {};
  }

  /// @brief shortcut getter for net
  container::ObserverPtr<Net> net(IndexType i) {
    if (netlist_) {
      return netlist_->net(i);
    }
    return {};
  }

  /// @brief shortcut getter for pin
  container::ObserverPtr<const Pin> pin(IndexType i) const {
    if (netlist_) {
      return netlist_->pin(i);
    }
    return {};
  }

  /// @brief shortcut getter for pin
  container::ObserverPtr<Pin> pin(IndexType i) {
    if (netlist_) {
      return netlist_->pin(i);
    }
    return {};
  }

  /// @brief add an instance
  Inst                              &tryAddInst(std::string const &name);

  /// @brief get the index of an instance, given instance name
  IndexType                          instId(std::string const &name) const;

  /// @brief get the pointer of an instance, given instance name
  container::ObserverPtr<const Inst> inst(std::string const &name) const;

  /// @brief get the pointer of an instance, given instance name
  container::ObserverPtr<Inst>       inst(std::string const &name);

  /// @brief add a net
  Net                               &tryAddNet(std::string const &name);

  /// @brief get the index of an net, given net name
  IndexType                          netId(std::string const &name) const;

  /// @brief get the pointer of an net, given net name
  container::ObserverPtr<const Net>  net(std::string const &name) const;

  /// @brief get the pointer of an net, given net name
  container::ObserverPtr<Net>        net(std::string const &name);

  /// @brief add a pin
  Pin                               &addPin();

  /// @brief summarize memory usage of the object in bytes
  IndexType                          memory() const;

 protected:
  /// @brief copy object
  void                                       copy(Model const &rhs);
  /// @brief move object
  void                                       move(Model &&rhs);
  /// @brief overload output stream
  friend std::ostream                       &operator<<(std::ostream &os, Model const &rhs);

  ModelType                                  model_type_;              ///< model type, Cell, IOPin, Module
  SizeType                                   size_;                    ///< physical size of the model
  std::string                                name_;                    ///< model name
  std::vector<ModelPin>                      model_pins_;              ///< IO pins for this model
  std::unordered_map<std::string, IndexType> model_pin_name2id_map_;   ///< map model pin name to index
  /// only for module
  boost::optional<NetlistType>               netlist_;   ///< internal circuit for the module
  boost::optional<std::unordered_map<IndexType, IndexType>>
          net_to_model_pin_;   // if the module has io pins, map some nets to these io pins.
  boost::optional<std::unordered_map<IndexType, IndexType>>   model_pin_to_net_;
  boost::optional<std::unordered_map<std::string, IndexType>> inst_name2id_map_;   ///< map instance name to index
  boost::optional<std::unordered_map<std::string, IndexType>> net_name2id_map_;    ///< map net name to index
};

}   // namespace database

OPENPARF_END_NAMESPACE

#endif
