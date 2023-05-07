/**
 * @file   dct2_fft2_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, modified by Yibo Lin
 * @date   Apr 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#include "ops/dct/src/dct2_fft2_cuda.h"

OPENPARF_BEGIN_NAMESPACE

void dct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                 at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CUDA(x);
  CHECK_FLAT_CUDA(expkM);
  CHECK_FLAT_CUDA(expkN);
  CHECK_FLAT_CUDA(out);
  CHECK_FLAT_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "dct2Forward", [&] {
    dct2PreprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N);

    buf = at::rfft(out, 2, false, true);

    dct2PostprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t));
  });
}

void idct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                  at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CUDA(x);
  CHECK_FLAT_CUDA(expkM);
  CHECK_FLAT_CUDA(expkN);
  CHECK_FLAT_CUDA(out);
  CHECK_FLAT_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idct2Forward", [&] {
    idct2PreprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idct2PostprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idctIdxstForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CUDA(x);
  CHECK_FLAT_CUDA(expkM);
  CHECK_FLAT_CUDA(expkN);
  CHECK_FLAT_CUDA(out);
  CHECK_FLAT_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idctIdxstForward", [&] {
    idctIdxstPreprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idctIdxstPostprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idxstIdctForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CUDA(x);
  CHECK_FLAT_CUDA(expkM);
  CHECK_FLAT_CUDA(expkN);
  CHECK_FLAT_CUDA(out);
  CHECK_FLAT_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idxstIdctForward", [&] {
    idxstIdctPreprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idxstIdctPostprocessCudaLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2", &OPENPARF_NAMESPACE::dct2Forward, "Dct2 with FFT2D (CUDA)");
  m.def("idct2", &OPENPARF_NAMESPACE::idct2Forward, "Idct2 FFT2D (CUDA)");
  m.def("idctIdxst", &OPENPARF_NAMESPACE::idctIdxstForward,
        "Idct Idxst FFT2D (CUDA)");
  m.def("idxstIdct", &OPENPARF_NAMESPACE::idxstIdctForward,
        "Idxst Idct FFT2D (CUDA)");
}
