/**
 * File              : util.h
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 10.24.2020
 * Last Modified Date: 10.24.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */

#ifndef OPENPARF_UTIL_UTIL_H_
#define OPENPARF_UTIL_UTIL_H_

#include <cxxabi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util/data_traits.hpp"
#include "util/enums.h"
#include "util/extensible_enum.h"
#include "util/map-macro.h"
#include "util/message.h"

OPENPARF_BEGIN_NAMESPACE

#define DISABLE_COPY_AND_ASSIGN(ClassName)                                                                             \
  ClassName(const ClassName &)            = delete;                                                                    \
  ClassName &operator=(const ClassName &) = delete;

#define DEFER_CONCAT_IMPL(x, y) x##y
#define DEFER_CONCAT(x, y)      DEFER_CONCAT_IMPL(x, y)

class DeferredAction {
  DISABLE_COPY_AND_ASSIGN(DeferredAction);

 public:
  explicit DeferredAction(std::function<void()> f) : f_(std::move(f)) {}
  ~DeferredAction() { f_(); }

 private:
  std::function<void()> f_;
};

#define DEFER(action) DeferredAction DEFER_CONCAT(__deferred__, __LINE__)([&]() -> void action)

template<typename T>
inline std::string className() {
  int32_t     status;
  const char *demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  return demangled;
}

template<typename T, typename Func>
inline void foreach2D(T xl, T yl, T xh, T yh, Func func) {
  for (auto x = xl; x < xh; ++x) {
    for (auto y = yl; y < yh; ++y) {
      func(x, y);
    }
  }
}

/// @brief Use this like a pointer with nullptr,
/// which is safe for python binding as well.
// template <typename T> using ObserverPtr = boost::optional<T &>;

#define ENUM_CASE(name)                                                                                           \
  case name:                                                                                                           \
    return #name;


#define MAKE_ENUM(type, ...)                                                                                           \
  enum type { __VA_ARGS__, unknown};                                                                                  \
  std::string toString(type e) {                                                                                       \
    switch (e) {                                                                                                       \
      MAP(ENUM_CASE, __VA_ARGS__)                                                                                 \
      default:                                                                                                         \
        return "unknown";                                                                                              \
    }                                                                                                                  \
  }


OPENPARF_END_NAMESPACE

#endif   // OPENPARF_UTIL_UTIL_H_
