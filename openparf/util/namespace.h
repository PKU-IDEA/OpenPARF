/**
 * @file   namespace.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_UTIL_NAMESPACE_H_
#define OPENPARF_UTIL_NAMESPACE_H_

#define OPENPARF_NAMESPACE openparf
#define OPENPARF_BEGIN_NAMESPACE namespace openparf {
#define OPENPARF_END_NAMESPACE }

/// @brief Force a function to be not inline.
/// This is compiler dependent.
#define OPENPARF_NOINLINE __attribute__((noinline))

/// @brief Differentiate math functions for CPU and CUDA code
#ifndef __NVCC__
#define OPENPARF_STD_NAMESPACE std
#define OPENPARF_HOST_DEVICE_FUNCTION
#else
#define OPENPARF_STD_NAMESPACE
#define OPENPARF_HOST_DEVICE_FUNCTION __host__ __device__
#endif

#endif
