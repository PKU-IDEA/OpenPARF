/**
 * @file   torch.h
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef OPENPARF_UTIL_TORCH_H_
#define OPENPARF_UTIL_TORCH_H_

/// As torch may change the header inclusion conventions,
/// it is better to manage it in a consistent way.
#if TORCH_MAJOR_VERSION >= 1
#include <torch/extension.h>

#if TORCH_MINOR_VERSION >= 3
#define OPENPARF_TENSOR_DATA_PTR(TENSOR, TYPE) TENSOR.data_ptr<TYPE>()
#define OPENPARF_TENSOR_SCALARTYPE(TENSOR) TENSOR.scalar_type()
#else
#define OPENPARF_TENSOR_DATA_PTR(TENSOR, TYPE) TENSOR.data<TYPE>()
#define OPENPARF_TENSOR_SCALARTYPE(TENSOR) TENSOR.type().scalarType()
#endif
#else
#include <torch/torch.h>
#endif
#include <limits>

#define CHECK_FLAT_CPU(x)                                                      \
  AT_ASSERTM(!x.is_cuda(), #x "must be a flat tensor on CPU")
#define CHECK_FLAT_CUDA(x)                                                     \
  AT_ASSERTM(x.is_cuda(), #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x)                                                          \
  AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_DIVISIBLE(x, y)                                                  \
  AT_ASSERTM((x.numel() % y) == 0, #x "must have "#y"-divisible number of elements")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// As the API for torch changes, customize a DREAMPlace version to remove
/// warnings
#define OPENPARF_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                      \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    (void)the_type;                                                            \
    at::ScalarType _st = OPENPARF_TENSOR_SCALARTYPE(TYPE);                     \
    switch (_st) {                                                             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)          \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");           \
    }                                                                          \
  }()

#endif
