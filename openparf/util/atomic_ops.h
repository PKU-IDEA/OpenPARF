/**
 * @file   atomic_ops.h
 * @author Yibo Lin
 * @date   Apr 2020
 */
#pragma once
#include <algorithm>
#include <type_traits>

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief A class generalized scaled atomic addition for floating point number
/// and integers. For integer, we use it as a fixed point number with the LSB
/// part for fractions.
template<typename T, bool = std::is_integral<T>::value>
struct AtomicAdd {
  typedef T type;

  /// @brief constructor
  explicit AtomicAdd(type = 1) {}

  template<typename V>
  inline void operator()(type* dst, V v) const {
#pragma omp atomic
    *dst += v;
  }
};

/// @brief For atomic addition of fixed point number using integers.
template<typename T>
struct AtomicAdd<T, true> {
  typedef T type;

  type      scale_factor;   ///< a scale factor to scale fraction into integer

  /// @brief constructor
  /// @param sf scale factor
  explicit AtomicAdd(type sf = 1) : scale_factor(sf) {}

  inline type scaleFactor() const { return scale_factor; }

  template<typename V>
  inline void operator()(type* dst, V v) const {
    type sv = v * scale_factor;
#pragma omp atomic
    *dst += sv;
  }
};

template<typename T, typename U>
inline void copyScaleArray(T* dst, U* src, T scale_factor, int n, int num_threads) {
  int32_t chunk_size = OPENPARF_STD_NAMESPACE::max(int32_t(n / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(static, chunk_size)
  for (int i = 0; i < n; i++) {
    dst[i] = src[i] * scale_factor;
  }
}

OPENPARF_END_NAMESPACE
