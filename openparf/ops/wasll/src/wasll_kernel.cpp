/**
 * File              : wasll_kernel.cpp
 * Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
 * Date              : 11.17.2023
 * Last Modified Date: 11.17.2023
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Compute weighted average super long lines and gradient.
///
/// @param pos (x, y) location of pins, dimension of (#pins, 2).
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin. The length is number of nets. The last entry equals to the
/// number of pins.
/// @param net_mask whether compute the wirelength for a net or not
/// @param net_weights weight of nets
/// @param num_nets number of nets.
/// @param num_slrX number of super logic regions along the X axis.
/// @param num_slrY number of super logic regions along the Y axis.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @param partial_wasll wirelength in x and y directions of each net. The first
/// half is the wirelength in x direction, and the second half is the wirelength
/// in y direction.
/// @param grad_intermediate back-propagated gradient from previous stage.
template <typename T>
void computeWASLL(T const *pos, int32_t const *flat_netpin,
                  int32_t const *netpin_start, uint8_t const *net_mask,
                  int32_t num_nets, int32_t num_slrX, int32_t num_slrY,
                  T const *inv_gamma, T *partial_wasll, T *grad_intermediate,
                  int32_t num_threads) {
  int32_t chunk_size = std::max(int32_t(num_nets / num_threads / 16), 1);

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int32_t i = 0; i < num_nets; ++i) {
    if (!net_mask[i]) continue;

    T x_max = -std::numeric_limits<T>::max();
    T x_min = std::numeric_limits<T>::max();
    T y_max = -std::numeric_limits<T>::max();
    T y_min = std::numeric_limits<T>::max();

    int32_t bgn = netpin_start[i];
    int32_t end = netpin_start[i + 1];
    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = flat_netpin[j] << 1;
      T       xx     = pos[offset];
      T       yy     = pos[offset + 1];
      x_max          = std::max(xx, x_max);
      x_min          = std::min(xx, x_min);
      y_max          = std::max(yy, y_max);
      y_min          = std::min(yy, y_min);
    }

    // pre-calculated
    T exp_x_max = std::exp(-x_max * (*inv_gamma));
    T exp_x_min = std::exp(x_min * (*inv_gamma));
    T exp_y_max = std::exp(-y_max * (*inv_gamma));
    T exp_y_min = std::exp(y_min * (*inv_gamma));

    T xexp_x_sum  = 0;
    T xexp_nx_sum = 0;
    T exp_x_sum   = 0;
    T exp_nx_sum  = 0;

    T yexp_y_sum  = 0;
    T yexp_ny_sum = 0;
    T exp_y_sum   = 0;
    T exp_ny_sum  = 0;

    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = flat_netpin[j] << 1;
      T       xx     = pos[offset];
      T       yy     = pos[offset + 1];

      if (num_slrX > 1) {
        T exp_x  = std::exp(xx * (*inv_gamma)) * exp_x_max;
        T exp_nx = std::exp(-xx * (*inv_gamma)) * exp_x_min;

        xexp_x_sum += xx * exp_x;
        xexp_nx_sum += xx * exp_nx;
        exp_x_sum += exp_x;
        exp_nx_sum += exp_nx;
      }

      if (num_slrY > 1) {
        T exp_y  = std::exp(yy * (*inv_gamma)) * exp_y_max;
        T exp_ny = std::exp(-yy * (*inv_gamma)) * exp_y_min;

        yexp_y_sum += yy * exp_y;
        yexp_ny_sum += yy * exp_ny;
        exp_y_sum += exp_y;
        exp_ny_sum += exp_ny;
      }
    }

    T partial_x_wasll = 0;
    T partial_y_wasll = 0;
    if (num_slrX > 1) {
      partial_x_wasll = (xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum);
    }
    if (num_slrY > 1) {
      partial_y_wasll = (yexp_y_sum / exp_y_sum - yexp_ny_sum / exp_ny_sum);
    }

    partial_wasll[i] = partial_x_wasll + partial_y_wasll;

    for (int32_t j = bgn; j < end; ++j) {
      int32_t offset = flat_netpin[j] << 1;
      T       xx     = pos[offset];
      T       yy     = pos[offset + 1];

      T gradx = 0;
      T grady = 0;
      if (num_slrX > 1) {
        T exp_x  = std::exp(xx * (*inv_gamma)) * exp_x_max;
        T exp_nx = std::exp(-xx * (*inv_gamma)) * exp_x_min;

        T b_x  = (*inv_gamma) * (1 / exp_x_sum);
        T a_x  = (1 - b_x * xexp_x_sum) / exp_x_sum;
        T b_nx = -(*inv_gamma) * (1 / exp_nx_sum);
        T a_nx = (1 - b_nx * xexp_nx_sum) / exp_nx_sum;

        gradx = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
      }
      if (num_slrY > 1) {
        T exp_y  = std::exp(yy * (*inv_gamma)) * exp_y_max;
        T exp_ny = std::exp(-yy * (*inv_gamma)) * exp_y_min;

        T b_y  = (*inv_gamma) * (1 / exp_y_sum);
        T a_y  = (1 - b_y * yexp_y_sum) / exp_y_sum;
        T b_ny = -(*inv_gamma) * (1 / exp_ny_sum);
        T a_ny = (1 - b_ny * yexp_ny_sum) / exp_ny_sum;

        grady = (a_y + b_y * yy) * exp_y - (a_ny + b_ny * yy) * exp_ny;
      }

      grad_intermediate[offset]     = gradx;
      grad_intermediate[offset + 1] = grady;
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
void OPENPARF_NOINLINE computeWASLLLauncher(
    T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start,
    uint8_t const *net_mask, int32_t num_nets, int32_t num_slrX,
    int32_t num_slrY, T const *inv_gamma, T *partial_wasll,
    T *grad_intermediate, int32_t num_threads) {
  computeWASLL<T>(pos, flat_netpin, netpin_start, net_mask, num_nets, num_slrX,
                  num_slrY, inv_gamma, partial_wasll, grad_intermediate,
                  num_threads);
}

template <typename T>
void integrateNetWeights(int32_t const *flat_netpin,
                         int32_t const *netpin_start, uint8_t const *net_mask,
                         T const *net_weights, T *grad_tensor, int32_t num_nets,
                         int32_t num_threads) {
  int32_t chunk_size = std::max(int32_t(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int32_t net_id = 0; net_id < num_nets; ++net_id) {
    if (net_mask[net_id]) {
      T weight = net_weights[net_id];
      for (int32_t j = netpin_start[net_id]; j < netpin_start[net_id + 1];
           ++j) {
        int32_t pin_id = flat_netpin[j];
        int32_t offset = (pin_id << 1);
        grad_tensor[offset] *= weight;
        grad_tensor[offset + 1] *= weight;
      }
    }
  }
}

template <typename T>
void OPENPARF_NOINLINE integrateNetWeightsLauncher(
    int32_t const *flat_netpin, int32_t const *netpin_start,
    uint8_t const *net_mask, T const *net_weights, T *grad_tensor,
    int32_t num_nets, int32_t num_threads) {
  integrateNetWeights(flat_netpin, netpin_start, net_mask, net_weights,
                      grad_tensor, num_nets, num_threads);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                          \
  void instantiateComputeWASLLLauncher(                                      \
      T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start, \
      uint8_t const *net_mask, T const *net_weights, int32_t num_nets,       \
      int32_t num_slrX, int32_t num_slrY, T const *inv_gamma,                \
      T *partial_wasll, T *grad_intermediate, int32_t num_threads) {         \
    computeWASLLLauncher<T>(pos, flat_netpin, netpin_start, net_mask,        \
                            num_nets, num_slrX, num_slrY, inv_gamma,         \
                            partial_wasll, grad_intermediate, num_threads);  \
    integrateNetWeightsLauncher<T>(flat_netpin, netpin_start, net_mask,      \
                                   net_weights, grad_intermediate, num_nets, \
                                   num_threads);                             \
  }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
