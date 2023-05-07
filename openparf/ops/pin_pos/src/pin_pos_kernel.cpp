/**
 * File              : pin_pos_kernel.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 04.21.2020
 * Last Modified Date: 04.21.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Given cell locations, compute pin locations
/// @param x cell locations in x direction
/// @param y cell locations in y direction
/// @param pin_offset_x pin offset in x direction
/// @param pin_offset_y pin offset in y direction
/// @param pin2node_map map pin index to node index
/// @param node_pins map node index to pins
/// @param node_pins_start start index of node_pins for each
/// node
/// @param num_nodes number of nodes
/// @param num_pins number of pins
/// @param num_threads number of threads
/// @param pin_x pin positions in x direction
/// @param pin_y pin positions in y direction

template <typename T>
void computePinPos(T const* pos, T const* pin_offsets,
                   int32_t const* pin2node_map, int32_t num_pins,
                   int32_t num_threads, T* pin_pos) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < num_pins; ++i) {
    int32_t node_id = pin2node_map[i];
    int32_t pin_id_offset = (i << 1);
    int32_t node_id_offset = (node_id << 1);
    pin_pos[pin_id_offset] = pin_offsets[pin_id_offset] + pos[node_id_offset];
    pin_pos[pin_id_offset + 1] =
        pin_offsets[pin_id_offset + 1] + pos[node_id_offset + 1];
  }
}
template <typename T>
void OPENPARF_NOINLINE computePinPosLauncher(T const* pos, T const* pin_offsets,
                                             int32_t const* pin2node_map,
                                             int32_t num_pins,
                                             int32_t num_threads, T* pin_pos) {
  computePinPos(pos, pin_offsets, pin2node_map, num_pins, num_threads, pin_pos);
}

template <typename T>
void computePinPosGrad(T const* grad_outs, int32_t const* node_pins,
                       int32_t const* node_pins_start, int32_t num_nodes,
                       int32_t num_threads, T* grads) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < num_nodes; ++i) {
    int32_t bgn = node_pins_start[i];
    int32_t end = node_pins_start[i + 1];
    int32_t offset = (i << 1);
    T& gx = grads[offset];
    T& gy = grads[offset + 1];
    for (int32_t j = bgn; j < end; ++j) {
      int32_t pin_id = node_pins[j];
      int32_t pin_id_offset = (pin_id << 1);
      gx += grad_outs[pin_id_offset];
      gy += grad_outs[pin_id_offset + 1];
    }
  }
}

template<typename T>
void OPENPARF_NOINLINE computePinPosGradLauncher(T const *grad_outs, int32_t const *node_pins,
                                                 int32_t const *node_pins_start, int32_t num_nodes,
                                                 int32_t num_threads, T *grads) {
    computePinPosGrad(grad_outs, node_pins, node_pins_start, num_nodes, num_threads, grads);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void computePinPosLauncher<T>(T const *pos, T const *pin_offsets,                     \
                                           int32_t const *pin2node_map, int32_t num_pins,          \
                                           int32_t num_threads, T *pin_pos);                       \
    template void computePinPosGradLauncher<T>(T const *grad_outs, int32_t const *node_pins,       \
                                               int32_t const *node_pins_start, int32_t num_nodes,  \
                                               int32_t num_threads, T *grads);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE
