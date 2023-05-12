/**
 * File              : stable_div_cuda.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.29.2020
 * Last Modified Date: 04.29.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Perform stable division with divisor tensor containing zeros
/// The idea is to handle x / 0 = 0.
template <typename T>
void computeStableDivCudaLauncher(T const* x, T const* y, int32_t n, T* out);

template <typename T>
void computeStableDivGradCudaLauncher(T const* grad_outs, T const* x,
                                      T const* y, int32_t n, T* gradx,
                                      T* grady);

at::Tensor stableDivForward(at::Tensor x, at::Tensor y) {
  CHECK_FLAT_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_FLAT_CUDA(y);
  CHECK_CONTIGUOUS(y);

  auto out = at::empty_like(x);
  int32_t n = x.numel();

  OPENPARF_DISPATCH_FLOATING_TYPES(
      x, "computeStableDivCudaLauncher", [&] {
        computeStableDivCudaLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(y, scalar_t), n,
            OPENPARF_TENSOR_DATA_PTR(out, scalar_t));
      });

  return out;
}

std::vector<at::Tensor> stableDivBackward(at::Tensor grad_out, at::Tensor x,
                                          at::Tensor y) {
  CHECK_FLAT_CUDA(grad_out);
  CHECK_CONTIGUOUS(grad_out);

  auto gradx = at::empty_like(x);
  auto grady = at::empty_like(y);
  int32_t n = x.numel();

  OPENPARF_DISPATCH_FLOATING_TYPES(
      grad_out, "computeStableDivGradCudaLauncher", [&] {
        computeStableDivGradCudaLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(grad_out, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(y, scalar_t), n,
            OPENPARF_TENSOR_DATA_PTR(gradx, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(grady, scalar_t));
      });

  return {gradx, grady};
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::stableDivForward,
        "StableDiv forward (CUDA)");
  m.def("backward", &OPENPARF_NAMESPACE::stableDivBackward,
        "StableDiv backward (CUDA)");
}
