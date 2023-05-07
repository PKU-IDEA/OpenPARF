/**
 * @file   atomic_ops.cuh
 * @author Yibo Lin
 * @date   Oct 2019
 */
#pragma once
#include <type_traits>

#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

/// @brief A class generalized scaled atomic addition for floating point number and integers.
/// For integer, we use it as a fixed point number with the LSB part for fractions.
template<typename T, bool = std::is_integral<T>::value>
struct AtomicAdd {
  typedef T type;

  /// @brief constructor
  explicit AtomicAdd(type = 1) {}

  template<typename V>
  __device__ __forceinline__ type operator()(type* dst, V v) const {
    return atomicAdd(dst, (type) v);
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

  type scaleFactor() const { return scale_factor; }

  template<typename V>
  __device__ __forceinline__ type operator()(type* dst, V v) const {
    type sv = v * scale_factor;
    return atomicAdd(dst, sv);
  }
};

template<typename T, typename U>
__global__ void copyScaleArray(T* dst, U* src, T scale_factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i] * scale_factor;
  }
}

OPENPARF_END_NAMESPACE
