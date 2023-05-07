/**
 * File              : hpwl_kernel.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.21.2020
 * Last Modified Date: 04.21.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void computeHPWL(T const *pos, int32_t const *flat_netpin,
                 int32_t const *netpin_start, uint8_t const *net_mask,
                 int32_t num_nets, int32_t num_threads, T *partial_hpwl) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < num_nets; ++i) {
    T max_x = -std::numeric_limits<T>::max();
    T min_x = std::numeric_limits<T>::max();
    T max_y = -std::numeric_limits<T>::max();
    T min_y = std::numeric_limits<T>::max();

    if (net_mask[i]) {
      for (int32_t j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        int32_t offset = (flat_netpin[j] << 1);
        auto xx = pos[offset];
        auto yy = pos[offset + 1];
        min_x = std::min(min_x, xx);
        max_x = std::max(max_x, xx);
        min_y = std::min(min_y, yy);
        max_y = std::max(max_y, yy);
      }
      partial_hpwl[(i << 1)] = max_x - min_x;
      partial_hpwl[(i << 1) + 1] = max_y - min_y;
    }
  }
}

/// @brief It is necesssary to add another wrapper with the noinline attribute.
/// This is a compiler related behavior where GCC will generate symbols with
/// "._omp_fn" suffix for OpenMP functions, causing the mismatch between
/// function declaration and implementation. As a result, at runtime, the
/// symbols will be not found. As GCC may also automatically inline the
/// function, we need to add this guard to guarantee noinline.
template <typename T>
void OPENPARF_NOINLINE
computeHPWLLauncher(T const *pos, int32_t const *flat_netpin,
                    int32_t const *netpin_start, uint8_t const *net_mask,
                    int32_t num_nets, int32_t num_threads, T *partial_hpwl) {
  computeHPWL(pos, flat_netpin, netpin_start, net_mask, num_nets, num_threads,
              partial_hpwl);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                          \
  void instantiateComputeHPWLLauncher(                                       \
      T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start, \
      uint8_t const *net_mask, int32_t num_nets, int32_t num_threads,        \
      T *partial_hpwl) {                                                     \
    computeHPWLLauncher(pos, flat_netpin, netpin_start, net_mask, num_nets,  \
                        num_threads, partial_hpwl);                          \
  }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
