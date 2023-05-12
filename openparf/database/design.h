/**
 * @file   design.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_DESIGN_H_
#define OPENPARF_DATABASE_DESIGN_H_

#include "container/container.hpp"
// local dependency
#include "database/inst.h"
#include "database/model.h"
#include "database/module_inst.h"
#include "database/shape_constr.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

class Design : public Object {
 public:
  using BaseType = Object;
  /// @brief Instantiation of the design.
  /// All instances are stored in a flat manner with an additional hierarchical
  /// module tree to record the hierarchy.
  using NetlistType = Netlist<Inst, Net, Pin>;

  /// @brief default constructor
  Design() : BaseType() {
    top_module_inst_ = std::numeric_limits<IndexType>::max();
  }

  /// @brief constructor
  Design(IndexType id) : BaseType(id) {
    top_module_inst_ = std::numeric_limits<IndexType>::max();
  }

  /// @brief copy constructor
  Design(Design const &rhs) { copy(rhs); }

  /// @brief move constructor
  Design(Design &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  Design &operator=(Design const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Design &operator=(Design &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief create model
  Model &addModel(std::string const &name);

  /// @brief number of models
  IndexType numModels() const { return models_.size(); }

  /// @brief getter for model
  Model const &model(IndexType i) const { return models_[i]; }

  /// @brief getter for model
  Model &model(IndexType i) { return models_[i]; }

  /// @brief getter for models
  std::vector<Model> const &models() const { return models_; }

  /// @brief getter for model index given model name;
  /// return infinity if not found
  IndexType modelId(std::string const &name) const;

  /// @brief getter for impl
  boost::optional<NetlistType> const &netlist() const { return netlist_; }

  /// @brief getter for impl
  IndexType VddVssNetId() const {return vddvss_net_id_;}

  /// @brief setter for impl
  void setNetlist(NetlistType const &rhs) { netlist_ = rhs; }

  /// @brief setter for impl
  void setNetlist(NetlistType &&rhs) { netlist_ = std::move(rhs); }

  /// @brief number of module instances
  IndexType numModuleInsts() const {
    if (module_insts_) {
      return module_insts_->size();
    }
    return 0;
  }

  /// @brief getter for module instance
  container::ObserverPtr<const ModuleInst> moduleInst(IndexType i) const;

  /// @brief getter for module instance
  container::ObserverPtr<ModuleInst> moduleInst(IndexType i);

  /// @brief getter for module instances
  boost::optional<std::vector<ModuleInst>> const &moduleInsts() const {
    return module_insts_;
  }

  /// @brief getter for top module index
  IndexType topModuleInstId() const { return top_module_inst_; }

  /// @brief setter for top module index
  void setTopModuleInstId(IndexType v) {
    top_module_inst_ = v;
  }

  /// @brief getter for top module
  container::ObserverPtr<const ModuleInst> topModuleInst() const;

  /// @brief getter for top module
  container::ObserverPtr<ModuleInst> topModuleInst();

  /// @brief number of shape constraints
  IndexType numShapeConstrs() const { return shape_constrs_.size(); }

  /// @brief getter for shape constraint
  ShapeConstr const &shapeConstr(IndexType i) const {
    openparfAssert(i < numShapeConstrs());
    return shape_constrs_[i];
  }

  /// @brief getter for shape constraint
  ShapeConstr &shapeConstr(IndexType i) {
    openparfAssert(i < numShapeConstrs());
    return shape_constrs_[i];
  }

  /// @brief add a shape constraint
  ShapeConstr &addShapeConstr(ShapeConstrType const &t);

  /// @brief uniquify and instantiate the top module
  /// @return current top module instance
  IndexType uniquify(IndexType model_id);

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

 protected:
  /// @brief copy object
  void copy(Design const &rhs);
  /// @brief move object
  void move(Design &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Design const &rhs);

  /// @brief kernel function for uniquification;
  /// use recursive instantiation
  /// @param module_inst target module instance to uniquify
  void uniquifyKernel(ModuleInst *module_inst);

  /// @brief instantiate a model as a module instance.
  /// If the model is not a module, then not added.
  ModuleInst *tryAddModuleInst(Model const &model);

  std::vector<Model>
      models_;  ///< all models, including the derived models, the
                ///< storage is here; not uniquified yet.
  std::unordered_map<std::string, IndexType>
      model_name2id_map_;  ///< map model name to index

  boost::optional<NetlistType>
      netlist_;  ///< flat instantiation of the design after uniquification

  IndexType vddvss_net_id_;  ///< net id of the VDD/VSS net

  boost::optional<std::vector<ModuleInst>>
      module_insts_;           ///< all module instances in hierarchy
  IndexType top_module_inst_;  ///< id of top module instance

  std::vector<ShapeConstr>
      shape_constrs_;  ///< shape constraints to describe
                       ///< sets of cells must form a pattern
};

}  // namespace database

OPENPARF_END_NAMESPACE

#endif
