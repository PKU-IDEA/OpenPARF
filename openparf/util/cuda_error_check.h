/**
 * File              : cuda_error_check.h
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.15.2021
 * Last Modified Date: 09.15.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 * Credit to: https://gist.github.com/InnovArul/68a720d6843aab6e246a
 */
#ifndef OPENPARF_UTIL_CUDA_ERROR_CHECK_H_
#define OPENPARF_UTIL_CUDA_ERROR_CHECK_H_

#include <cstdio>

#include "cuda_runtime.h"

// enable error check
#define CUDA_ERROR_CHECK

// check the synchronous function call errorcode 'err' if it is a cudaSuccess
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

// check if any error happened during asynchronous execution of Cuda kernel __global__ function
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)


/**
 * API to call Cuda APIs safely
 * @param err  - error code to be checked
 * @param file - file name
 * @param line - line number
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

/**
 * API to check the last returned cuda error
 * @param file - filename
 * @param line - line number
 */
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

#endif   // OPENPARF_UTIL_CUDA_ERROR_CHECK_H_
