/**
 * @file   dct2_fft2.cpp
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#include "ops/dct/src/dct2_fft2.h"

OPENPARF_BEGIN_NAMESPACE

void dct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                 at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CPU(x);
  CHECK_FLAT_CPU(expkM);
  CHECK_FLAT_CPU(expkN);
  CHECK_FLAT_CPU(out);
  CHECK_FLAT_CPU(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "dct2Forward", [&] {
    dct2dPreprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N, at::get_num_threads());

    buf = at::rfft(out, 2, false, true);

    dct2dPostprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t), at::get_num_threads());
  });
}

void idct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                  at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CPU(x);
  CHECK_FLAT_CPU(expkM);
  CHECK_FLAT_CPU(expkN);
  CHECK_FLAT_CPU(out);
  CHECK_FLAT_CPU(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idct2Forward", [&] {
    idct2PreprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t), at::get_num_threads());

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idct2PostprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N, at::get_num_threads());
  });
}

void idctIdxstForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CPU(x);
  CHECK_FLAT_CPU(expkM);
  CHECK_FLAT_CPU(expkN);
  CHECK_FLAT_CPU(out);
  CHECK_FLAT_CPU(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idctIdxstForward", [&] {
    idctIdxstPreprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t), at::get_num_threads());

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idctIdxstPostprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N, at::get_num_threads());
  });
}

void idxstIdctForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf) {
  CHECK_FLAT_CPU(x);
  CHECK_FLAT_CPU(expkM);
  CHECK_FLAT_CPU(expkN);
  CHECK_FLAT_CPU(out);
  CHECK_FLAT_CPU(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  OPENPARF_DISPATCH_FLOATING_TYPES(x, "idxstIdctForward", [&] {
    idxstIdctPreprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(x, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        OPENPARF_TENSOR_DATA_PTR(expkM, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(expkN, scalar_t), at::get_num_threads());

    auto y = at::irfft(buf, 2, false, true, {M, N});

    idxstIdctPostprocessCpuLauncher<scalar_t>(
        OPENPARF_TENSOR_DATA_PTR(y, scalar_t),
        OPENPARF_TENSOR_DATA_PTR(out, scalar_t), M, N, at::get_num_threads());
  });
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2", &OPENPARF_NAMESPACE::dct2Forward, "Dct2 with FFT2D");
  m.def("idct2", &OPENPARF_NAMESPACE::idct2Forward, "Idct2 with FFT2D");
  m.def("idctIdxst", &OPENPARF_NAMESPACE::idctIdxstForward,
        "Idct Idxst with FFT2D");
  m.def("idxstIdct", &OPENPARF_NAMESPACE::idxstIdctForward,
        "Idxst Idct with FFT2D");
}
