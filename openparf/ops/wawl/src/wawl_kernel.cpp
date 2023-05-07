/**
 * @file   wawl_kernel.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*std::exp(x_i/gamma) / \sum_i std::exp(x_i/gamma) - \sum_i
/// x_i*std::exp(-x_i/gamma) / \sum_i x_i*std::exp(-x_i/gamma), where x_i is pin
/// location.
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
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first
/// half is the wirelength in x direction, and the second half is the wirelength
/// in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T>
void computeWAWL(T const *pos, int32_t const *flat_netpin,
                 int32_t const *netpin_start, uint8_t const *net_mask,
                 int32_t num_nets, T const *inv_gamma, T *partial_wl,
                 T *grad_intermediate, int32_t num_threads) {
  int32_t chunk_size = std::max(int32_t(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int32_t i = 0; i < num_nets; ++i) {
    if (net_mask[i]) {
      T x_max = -std::numeric_limits<T>::max();
      T x_min = std::numeric_limits<T>::max();
      T y_max = -std::numeric_limits<T>::max();
      T y_min = std::numeric_limits<T>::max();
      int32_t bgn = netpin_start[i];
      int32_t end = netpin_start[i + 1];
      for (int32_t j = bgn; j < end; ++j) {
        int32_t offset = (flat_netpin[j] << 1);
        T xx = pos[offset];
        T yy = pos[offset + 1];
        x_max = std::max(xx, x_max);
        x_min = std::min(xx, x_min);
        y_max = std::max(yy, y_max);
        y_min = std::min(yy, y_min);
      }

      T xexp_x_sum = 0;
      T xexp_nx_sum = 0;
      T exp_x_sum = 0;
      T exp_nx_sum = 0;

      T yexp_y_sum = 0;
      T yexp_ny_sum = 0;
      T exp_y_sum = 0;
      T exp_ny_sum = 0;

      for (int32_t j = bgn; j < end; ++j) {
        int32_t offset = (flat_netpin[j] << 1);
        T xx = pos[offset];
        T yy = pos[offset + 1];

        T exp_x = std::exp((xx - x_max) * (*inv_gamma));
        T exp_nx = std::exp((x_min - xx) * (*inv_gamma));

        xexp_x_sum += xx * exp_x;
        xexp_nx_sum += xx * exp_nx;
        exp_x_sum += exp_x;
        exp_nx_sum += exp_nx;

        T exp_y = std::exp((yy - y_max) * (*inv_gamma));
        T exp_ny = std::exp((y_min - yy) * (*inv_gamma));

        yexp_y_sum += yy * exp_y;
        yexp_ny_sum += yy * exp_ny;
        exp_y_sum += exp_y;
        exp_ny_sum += exp_ny;
      }

      partial_wl[i] = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum +
                      yexp_y_sum / exp_y_sum - yexp_ny_sum / exp_ny_sum;

      T b_x = (*inv_gamma) / (exp_x_sum);
      T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
      T b_nx = -(*inv_gamma) / (exp_nx_sum);
      T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

      T b_y = (*inv_gamma) / (exp_y_sum);
      T a_y = (1.0 - b_y * yexp_y_sum) / exp_y_sum;
      T b_ny = -(*inv_gamma) / (exp_ny_sum);
      T a_ny = (1.0 - b_ny * yexp_ny_sum) / exp_ny_sum;

      for (int32_t j = bgn; j < end; ++j) {
        int32_t offset = (flat_netpin[j] << 1);
        T xx = pos[offset];
        T yy = pos[offset + 1];
        auto &gradx = grad_intermediate[offset];
        auto &grady = grad_intermediate[offset + 1];

        T exp_x = std::exp((xx - x_max) * (*inv_gamma));
        T exp_nx = std::exp((x_min - xx) * (*inv_gamma));

        gradx = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

        T exp_y = std::exp((yy - y_max) * (*inv_gamma));
        T exp_ny = std::exp((y_min - yy) * (*inv_gamma));

        grady = (a_y + b_y * yy) * exp_y - (a_ny + b_ny * yy) * exp_ny;
      }
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
void OPENPARF_NOINLINE computeWAWLLauncher(
    T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start,
    uint8_t const *net_mask, int32_t num_nets, T const *inv_gamma,
    T *partial_wl, T *grad_intermediate, int32_t num_threads) {
  computeWAWL(pos, flat_netpin, netpin_start, net_mask, num_nets, inv_gamma,
              partial_wl, grad_intermediate, num_threads);
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
#define REGISTER_KERNEL_LAUNCHER(T)                                            \
  void instantiateComputeWAWLLauncher(                                         \
      T const *pos, int32_t const *flat_netpin, int32_t const *netpin_start,   \
      uint8_t const *net_mask, T const *net_weights, int32_t num_nets,         \
      T const *inv_gamma, T *partial_wl, T *grad_intermediate,                 \
      int32_t num_threads) {                                                   \
    computeWAWLLauncher<T>(pos, flat_netpin, netpin_start, net_mask, num_nets, \
                           inv_gamma, partial_wl, grad_intermediate,           \
                           num_threads);                                       \
    integrateNetWeightsLauncher<T>(flat_netpin, netpin_start, net_mask,        \
                                   net_weights, grad_intermediate, num_nets,   \
                                   num_threads);                               \
  }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
