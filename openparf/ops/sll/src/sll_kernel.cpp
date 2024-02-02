/**
 * File              : sll_kernel.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void        computeSLLCounts(const T* pos, const int32_t* flat_netpin,
                             const int32_t* netpin_start, const uint8_t* net_mask,
                             const int32_t* sll_counts_table, int32_t* partial_sll,
                             int32_t num_nets, T xl, T yl, T slr_width, T slr_height,
                             int32_t num_slrX, int32_t num_slrY, int32_t num_threads) {
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
  for (int32_t i = 0; i < num_nets; ++i) {
    auto&                   sll                = partial_sll[i];
    const int32_t           num_slrs           = num_slrX * num_slrY;
    const int32_t           pin_num            = netpin_start[i + 1] - netpin_start[i];
    const constexpr int32_t net_ignore_pin_num = 2000;
    if (net_mask[i] || pin_num < net_ignore_pin_num) {
      // Initialize an array to record the distribution of net pins across the SLRs
      int32_t slr_dist[4]    = {0};
      int32_t temp           = 0;
      T       inv_slr_width  = 1.0 / slr_width;
      T       inv_slr_height = 1.0 / slr_height;

      for (int32_t j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        int32_t offset = flat_netpin[j] << 1;
        int32_t xx     = static_cast<int32_t>((pos[offset] - xl) * inv_slr_width);
        int32_t yy =
            static_cast<int32_t>((pos[offset + 1] - yl) * inv_slr_height);
        slr_dist[xx * num_slrY + yy] = 1;
      }
      for (int32_t idx = 0; idx < num_slrs; idx++) {
        temp |= slr_dist[idx] << (num_slrs - idx - 1);
      }
      sll = sll_counts_table[temp];
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
void OPENPARF_NOINLINE computeSLLLauncher(
    const T* pos, const int32_t* flat_netpin, const int32_t* netpin_start,
    const uint8_t* net_mask, int32_t* sll_counts_table, int32_t* partial_sll,
    int32_t num_nets, T xl, T yl, T slr_width, T slr_height, int32_t num_slrX,
    int32_t num_slrY, int32_t num_threads) {
  computeSLLCounts<T>(pos, flat_netpin, netpin_start, net_mask,
                      sll_counts_table, partial_sll, num_nets, xl, yl,
                      slr_width, slr_height, num_slrX, num_slrY, num_threads);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                             \
  void instantiatecomputeSLLLauncher(                                           \
      T const* pos, int32_t const* flat_netpin, int32_t const* netpin_start,    \
      uint8_t const* net_mask, int32_t* sll_counts_table,                       \
      int32_t* partial_sll, int32_t num_nets, T xl, T yl, T slr_width,          \
      T slr_height, int32_t num_slrX, int32_t num_slrY, int32_t num_threads) {  \
    computeSLLLauncher(pos, flat_netpin, netpin_start, net_mask,                \
                       sll_counts_table, partial_sll, num_nets, xl, yl,         \
                       slr_width, slr_height, num_slrX, num_slrY, num_threads); \
  }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE