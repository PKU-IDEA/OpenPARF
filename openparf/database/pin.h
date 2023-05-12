/**
 * @file   pin.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_PIN_H_
#define OPENPARF_DATABASE_PIN_H_

#include "database/attr.h"
#include "database/attr_object.hpp"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Pin for cell IO.
class ModelPin : public Object {
public:
  using BaseType = Object;
  using PointType = geometry::Point<CoordinateType, 2>;

  /// @brief default constructor
  ModelPin()
      : BaseType(), name_(), offset_(0, 0),
        signal_direct_(SignalDirection::kUnknown) {}

  /// @brief constructor
  ModelPin(IndexType id, std::string const &name)
      : BaseType(id), name_(name), offset_(0, 0),
        signal_direct_(SignalDirection::kUnknown) {}

  /// @brief copy constructor
  ModelPin(ModelPin const &rhs) { copy(rhs); }

  /// @brief move constructor
  ModelPin(ModelPin &&rhs) noexcept { move(std::move(std::move(rhs))); }

  /// @brief copy assignment
  ModelPin &operator=(ModelPin const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  ModelPin &operator=(ModelPin &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief getter for name
  std::string const &name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &n) { name_ = n; }

  /// @brief getter for offset
  PointType const &offset() const { return offset_; }

  /// @brief setter for offset
  void setOffset(PointType const &o) { offset_ = o; }

  /// @brief getter for signal direct
  SignalDirection const &signalDirect() const { return signal_direct_; }

  /// @brief setter for signal direct
  void setSignalDirect(SignalDirection const &o) { signal_direct_ = o; }

  /// @brief getter for signal type
  SignalType const &signalType() const { return signal_type_; }

  /// @brief setter for signal type
  void setSignalType(SignalType const &o) { signal_type_ = o; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(ModelPin const &rhs);
  /// @brief move object
  void move(ModelPin &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, ModelPin const &rhs);

  std::string name_; ///< pin name
  PointType offset_; ///< offset to cell origin, assume the origin is (0, 0)
  SignalDirection signal_direct_; ///< signal direction, e.g., input, output
  SignalType signal_type_;        ///< signal type, e.g., clock, control, signal
};

/// @brief Pin represents the connection point
/// in a graph. An instance has multiple pins
/// and a net has multiple pins as well.
class Pin : public AttrObject<PinAttr> {
public:
  using BaseType = AttrObject<PinAttr>;
  using AttrType = BaseType::AttrType;

  /// @brief default constructor
  Pin()
      : BaseType(), inst_id_(std::numeric_limits<IndexType>::max()),
        net_id_(std::numeric_limits<IndexType>::max()) {}

  /// @brief constructor
  Pin(IndexType id)
      : BaseType(id), inst_id_(std::numeric_limits<IndexType>::max()),
        net_id_(std::numeric_limits<IndexType>::max()) {}

  /// @brief copy constructor
  Pin(Pin const &rhs) { copy(rhs); }

  /// @brief move constructor
  Pin(Pin &&rhs) noexcept { move(std::move(std::move(rhs))); }

  /// @brief copy assignment
  Pin &operator=(Pin const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  Pin &operator=(Pin &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(std::move(rhs)));
    }
    return *this;
  }

  /// @brief getter for model pin index
  IndexType modelPinId() const { return model_pin_id_; }

  /// @brief setter for model pin index
  void setModelPinId(IndexType const &v) { model_pin_id_ = v; }

  /// @brief getter for instance index
  IndexType instId() const { return inst_id_; }

  /// @brief setter for instance index
  void setInstId(IndexType const &v) { inst_id_ = v; }

  /// @brief getter for net
  IndexType netId() const { return net_id_; }

  /// @brief setter for net
  void setNetId(IndexType const &v) { net_id_ = v; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(Pin const &rhs);
  /// @brief move object
  void move(Pin &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, Pin const &rhs);

  IndexType model_pin_id_; ///< model pin index in the model for the instance
  IndexType inst_id_;      ///< an instance index of cell or IO, nullptr for IO
  IndexType net_id_;       ///< net index
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
