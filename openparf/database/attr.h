/**
 * @file   attr.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_ATTR_H_
#define OPENPARF_DATABASE_ATTR_H_

#include "database/object.h"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

class BaseAttr : public Object {
public:
  using BaseType = Object;

  /// @brief default constructor
  BaseAttr() : BaseType() {}

  /// @brief constructor
  BaseAttr(IndexType id) : BaseType(id) {}

  /// @brief copy constructor
  BaseAttr(BaseAttr const &rhs) { copy(rhs); }

  /// @brief move constructor
  BaseAttr(BaseAttr &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  BaseAttr &operator=(BaseAttr const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  BaseAttr &operator=(BaseAttr &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

protected:
  /// @brief copy object
  void copy(BaseAttr const &rhs);
  /// @brief move object
  void move(BaseAttr &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, BaseAttr const &rhs);
};

/// @brief Record all attributes for an instance
/// without any interconnect information.
class InstAttr : public BaseAttr {
public:
  using BaseType = BaseAttr;
  using SizeType = geometry::Size<IndexType, 2>;
  using PointType = geometry::Point<IndexType, 3>;

  /// @brief default constructor
  InstAttr() : BaseType() {
    model_id_ = std::numeric_limits<IndexType>::max();
    place_status_ = PlaceStatus::kMovable;
  }

  /// @brief constructor
  InstAttr(IndexType id) : BaseType(id) {
    model_id_ = std::numeric_limits<IndexType>::max();
    place_status_ = PlaceStatus::kMovable;
  }

  /// @brief copy constructor
  InstAttr(InstAttr const &rhs) { copy(rhs); }

  /// @brief move constructor
  InstAttr(InstAttr &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  InstAttr &operator=(InstAttr const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  InstAttr &operator=(InstAttr &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for model
  IndexType modelId() const { return model_id_; }
  /// @brief setter for model
  void setModelId(IndexType m) { model_id_ = m; }

  /// @brief getter for placement status
  PlaceStatus const &placeStatus() const { return place_status_; }

  /// @brief setter for placement status
  void setPlaceStatus(PlaceStatus const &v) { place_status_ = v; }

  /// @brief getter for location
  PointType const &loc() const { return loc_; }

  /// @brief setter for location
  void setLoc(PointType const &v) { loc_ = v; }

  /// @brief getter for name
  std::string const &name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &name) { name_ = name; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(InstAttr const &rhs);
  /// @brief move object
  void move(InstAttr &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, InstAttr const &rhs);

  IndexType model_id_;       ///< model type
  PlaceStatus place_status_; ///< movable or fixed status
  PointType loc_;            ///< location of the instance
  std::string name_;         ///< name of the instance
};

/// @brief Record all attributes for a net
/// without any interconnect information.
class NetAttr : public BaseAttr {
public:
  using BaseType = BaseAttr;
  using WeightType = CoordinateTraits<CoordinateType>::WeightType;

  /// @brief default constructor
  NetAttr() : BaseType(), weight_(1) {}

  /// @brief constructor
  NetAttr(IndexType id) : BaseType(id), weight_(1) {}

  /// @brief copy constructor
  NetAttr(NetAttr const &rhs) { copy(rhs); }

  /// @brief move constructor
  NetAttr(NetAttr &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  NetAttr &operator=(NetAttr const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  NetAttr &operator=(NetAttr &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for weight
  WeightType weight() const { return weight_; }

  /// @brief setter for weight
  void setWeight(WeightType w) { weight_ = w; }

  /// @brief getter for name
  std::string const &name() const { return name_; }

  /// @brief setter for name
  void setName(std::string const &name) { name_ = name; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(NetAttr const &rhs);
  /// @brief move object
  void move(NetAttr &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, NetAttr const &rhs);

  WeightType weight_; ///< net weight
  std::string name_;
};

/// @brief Record all attributes for a pin.
class PinAttr : public BaseAttr {
public:
  using BaseType = BaseAttr;

  /// @brief default constructor
  PinAttr() : BaseType(), signal_direct_(SignalDirection::kUnknown) {}

  /// @brief constructor
  PinAttr(IndexType id)
      : BaseType(id), signal_direct_(SignalDirection::kUnknown) {}

  /// @brief copy constructor
  PinAttr(PinAttr const &rhs) { copy(rhs); }

  /// @brief move constructor
  PinAttr(PinAttr &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  PinAttr &operator=(PinAttr const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  PinAttr &operator=(PinAttr &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for signal direction
  SignalDirection signalDirect() const { return signal_direct_; }

  /// @brief setter for signal direction
  void setSignalDirect(SignalDirection const &s) { signal_direct_ = s; }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const;

protected:
  /// @brief copy object
  void copy(PinAttr const &rhs);
  /// @brief move object
  void move(PinAttr &&rhs);
  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, PinAttr const &rhs);

  SignalDirection
      signal_direct_; ///< signal direction, e.g., input, output, etc.
};

} // namespace database

OPENPARF_END_NAMESPACE

#endif
