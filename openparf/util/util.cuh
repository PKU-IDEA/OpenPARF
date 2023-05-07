/**
 * @file   util.cuh
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Utilities for CUDA
 */

#ifndef OPENPARF_UTIL_UTIL_CUH_
#define OPENPARF_UTIL_UTIL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>

#include <cfloat>
#include <cmath>
#include <cstdio>

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief template specialization for non-integral types
template<typename T>
inline __host__ __device__ typename std::enable_if<!std::is_integral<T>::value, T>::type ceilDiv(T a, T b) {
  return a / b;
}

/// @brief template specialization for integral types
template<typename T>
inline __host__ __device__ typename std::enable_if<std::is_integral<T>::value, T>::type ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

#define allocateCUDA(var, size)                                                                                        \
  {                                                                                                                    \
    cudaError_t status = cudaMalloc(&(var), (size) * sizeof(typename std::remove_pointer<decltype(var)>::type));       \
    openparfAssertMsg(status == cudaSuccess, "cudaMalloc failed for " #var);                                           \
  }

#define destroyCUDA(var)                                                                                               \
  {                                                                                                                    \
    cudaError_t status = cudaFree(var);                                                                                \
    openparfAssertMsg(status == cudaSuccess, "cudaFree failed for " #var);                                             \
  }

#define memsetCUDA(var, value, size)                                                                                   \
  {                                                                                                                    \
    cudaError_t status = cudaMemset(var, value, (size) * sizeof(typename std::remove_pointer<decltype(var)>::type));   \
    openparfAssertMsg(status == cudaSuccess, "cudaMemset failed for " #var);                                           \
  }

#define checkCUDA(status)                                                                                              \
  { openparfAssertMsg(status == cudaSuccess, "CUDA Runtime Error %s", cudaGetErrorString(status)); }

#define allocateCopyCUDA(var, rhs, size)                                                                               \
  {                                                                                                                    \
    allocateCUDA(var, size);                                                                                           \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(typename std::remove_pointer<decltype(var)>::type) * (size),                 \
            cudaMemcpyHostToDevice));                                                                                  \
  }

#define allocateCopyHost(var, rhs, size)                                                                               \
  {                                                                                                                    \
    var = (typename std::remove_pointer<decltype(var)>::type *) malloc(                                                \
            (size) * sizeof(typename std::remove_pointer<decltype(var)>::type));                                       \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(typename std::remove_pointer<decltype(var)>::type) * (size),                 \
            cudaMemcpyDeviceToHost));                                                                                  \
  }

OPENPARF_END_NAMESPACE

#endif
