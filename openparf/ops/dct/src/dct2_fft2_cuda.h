/**
 * @file   dct2_fft2_cuda.h
 * @author Zixuan Jiang, Jiaqi Gu, modified by Yibo Lin
 * @date   Apr 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#ifndef OPENPARF_OPS_DCT_SRC_DCT2_FFT2_CUDA_H_
#define OPENPARF_OPS_DCT_SRC_DCT2_FFT2_CUDA_H_

#include "util/torch.h"
#include "util/util.h"
#include <float.h>
#include <math.h>

OPENPARF_BEGIN_NAMESPACE

// dct2 with fft2
void dct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                 at::Tensor out, at::Tensor buf);

template <typename T>
void dct2PreprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                int32_t const N);

template <typename T>
void dct2PostprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                 int32_t const N, T const *__restrict__ expkM,
                                 T const *__restrict__ expkN);

// idct2 with fft2
void idct2Forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                  at::Tensor out, at::Tensor buf);

template <typename T>
void idct2PreprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                 int32_t const N, T const *__restrict__ expkM,
                                 T const *__restrict__ expkN);

template <typename T>
void idct2PostprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                  int32_t const N);

// idct idxst
void idctIdxstForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf);

template <typename T>
void idctIdxstPreprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                     int32_t const N,
                                     T const *__restrict__ expkM,
                                     T const *__restrict__ expkN);

template <typename T>
void idctIdxstPostprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                      int32_t const N);

// idxst idct
void idxstIdctForward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                      at::Tensor out, at::Tensor buf);

template <typename T>
void idxstIdctPreprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                     int32_t const N,
                                     T const *__restrict__ expkM,
                                     T const *__restrict__ expkN);

template <typename T>
void idxstIdctPostprocessCudaLauncher(T const *x, T *y, int32_t const M,
                                      int32_t const N);

OPENPARF_END_NAMESPACE

#endif
