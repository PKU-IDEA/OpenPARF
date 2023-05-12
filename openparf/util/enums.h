/**
 * @file   enums.h
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef OPENPARF_UTIL_ENUMS_H_
#define OPENPARF_UTIL_ENUMS_H_

#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/message.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Types of models (stencil).
enum class ModelType : uint8_t { kCell, kIOPin, kModule, kUnknown };

inline std::string toString(ModelType const &v) {
  switch (v) {
    case ModelType::kCell:
      return "ModelType.kCell";
    case ModelType::kIOPin:
      return "ModelType.kIOPin";
    case ModelType::kModule:
      return "ModelType.kModule";
    default:
      return "ModelType.kUnknown";
  }
}

/// @brief Signal direction types.
enum class SignalDirection : uint8_t { kInput, kOutput, kInout, kUnknown };

inline std::string toString(SignalDirection const &v) {
  switch (v) {
    case SignalDirection::kInput:
      return "SignalDirection.kInput";
    case SignalDirection::kOutput:
      return "SignalDirection.kOutput";
    case SignalDirection::kInout:
      return "SignalDirection.kInout";
    default:
      return "SignalDirection.kUnknown";
  }
}

/// @brief Signal types.
enum class SignalType : uint8_t {
  kClock,       ///< clock signal
  kControlSR,   ///< control signal, set/reset
  kControlCE,   ///< control signal, clock enable
  kCascade,     ///< cascade signal
  kSignal,
  kUnknown
};

inline std::string toString(SignalType const &v) {
  switch (v) {
    case SignalType::kClock:
      return "SignalType.kClock";
    case SignalType::kControlSR:
      return "SignalType.kControlSR";
    case SignalType::kControlCE:
      return "SignalType.kControlCE";
    case SignalType::kCascade:
      return "SignalType.kCascade";
    case SignalType::kSignal:
      return "SignalType.kSignal";
    default:
      return "SignalType.kUnknown";
  }
}

/// Get the control pin type (PinType::SR or PinType::CE) for a given pin name
inline SignalType ctrlPinNameToPinType(const std::string &str) {
  if (str == "S" || str == "R" || str == "PRE" || str == "CLR") {
    return SignalType::kControlSR;
  }
  if (str == "CE" || str == "GE") {
    return SignalType::kControlCE;
  }
  openparfAssertMsg(false, "Unexpected control pin name '%s'.\n", str.c_str());
}


/// @brief shape constraint types
enum class ShapeConstrType : uint8_t { kDSPChain, kRAMChain, kCarryChain, kUnknown };

inline std::string toString(ShapeConstrType const &v) {
  switch (v) {
    case ShapeConstrType::kDSPChain:
      return "ShapeConstrType.kDSPChain";
    case ShapeConstrType::kRAMChain:
      return "ShapeConstrType.kRAMChain";
    case ShapeConstrType::kCarryChain:
      return "ShapeConstrType.kCarryChain";
    default:
      return "ShapeConstrType.kUnknown";
  }
}

/// @brief placement status
enum class PlaceStatus : uint8_t { kMovable, kFixed, kUnknown };

inline std::string toString(PlaceStatus const &v) {
  switch (v) {
    case PlaceStatus::kMovable:
      return "PlaceStatus.kMovable";
    case PlaceStatus::kFixed:
      return "PlaceStatus.kFixed";
    default:
      return "PlaceStatus.kUnknown";
  }
}

// Enum for resource types
enum class RsrcType : uint8_t { LUTL, LUTM, FF, DSP, RAM, LUTL_PAIR, FF_PAIR, BLEL, SLICEL, SLICEM, IO, INVALID };

/// @brief resource category
/// The category is constructed by the demand of algorithm,
/// where we need to differentiate some types.
enum class ResourceCategory : uint8_t {
  kLUTL,    ///< LUTL
  kLUTM,    ///< LUTM
  kFF,      ///< flip-flop
  kCarry,   ///< carry
  kSSSIR,   ///< single-site single-instance resource like DSP/RAM
  kSSMIR,   ///< single-site multiple-instance resource like IO
  kUnknown
};

inline std::string toString(ResourceCategory const &v) {
  switch (v) {
    case ResourceCategory::kLUTL:
      return "ResourceCategory.kLUTL";
    case ResourceCategory::kLUTM:
      return "ResourceCategory.kLUTM";
    case ResourceCategory::kFF:
      return "ResourceCategory.kFF";
    case ResourceCategory::kCarry:
      return "ResourceCategory.kCarry";
    case ResourceCategory::kSSSIR:
      return "ResourceCategory.kSSSIR";
    case ResourceCategory::kSSMIR:
      return "ResourceCategory.kSSMIR";
    default:
      return "ResourceCategory.kUnknown";
  }
}

/// @brief Explicitly convert enum type to underlying values
template<typename E>
constexpr inline typename std::enable_if<std::is_enum<E>::value, typename std::underlying_type<E>::type>::type
toInteger(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

/// @brief Explicitly convert an integer value to enum
template<typename E, typename T>
constexpr inline typename std::enable_if<std::is_enum<E>::value && std::is_integral<T>::value, E>::type toEnum(
        T value) noexcept {
  return static_cast<E>(value);
}

/// @brief Overload output stream for enum type
template<typename E>
inline typename std::enable_if<std::is_enum<E>::value, std::ostream &>::type operator<<(std::ostream &os, E e) {
  os << toString(e);
  return os;
}

OPENPARF_END_NAMESPACE

#endif
