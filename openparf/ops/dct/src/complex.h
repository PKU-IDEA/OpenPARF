/**
 * @file   complex.h
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  Complex number for CPU/CUDA
 */

#ifndef OPENPARF_OPS_DCT_SRC_COMPLEX_H_
#define OPENPARF_OPS_DCT_SRC_COMPLEX_H_

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T> struct Complex {
  T x;
  T y;
  OPENPARF_HOST_DEVICE_FUNCTION Complex() {
    x = 0;
    y = 0;
  }

  OPENPARF_HOST_DEVICE_FUNCTION Complex(T real, T imag) {
    x = real;
    y = imag;
  }

  OPENPARF_HOST_DEVICE_FUNCTION ~Complex() {}
};

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION Complex<T>
complexMul(Complex<T> const &x, Complex<T> const &y) {
  Complex<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION T RealPartOfMul(Complex<T> const &x,
                                                     Complex<T> const &y) {
  return x.x * y.x - x.y * y.y;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION T ImaginaryPartOfMul(Complex<T> const &x,
                                                          Complex<T> const &y) {
  return x.x * y.y + x.y * y.x;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION Complex<T>
complexAdd(Complex<T> const &x, Complex<T> const &y) {
  Complex<T> res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION Complex<T>
complexSubtract(Complex<T> const &x, Complex<T> const &y) {
  Complex<T> res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION Complex<T>
complexConj(Complex<T> const &x) {
  Complex<T> res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

template <typename T>
inline OPENPARF_HOST_DEVICE_FUNCTION Complex<T>
complexMulConj(Complex<T> const &x, Complex<T> const &y) {
  Complex<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

OPENPARF_END_NAMESPACE

#endif
