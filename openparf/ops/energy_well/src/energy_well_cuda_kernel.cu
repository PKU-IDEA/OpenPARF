/**
 * File              : energy_well_kernel_cuda.cu
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 05.12.2021
 * Last Modified Date: 01.10.2024
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

/**
 * @brief Compute the integral of piecewise power function f(x).
 * @details The piecewise power function is \f{equation*}{
 * f(x) = \left\{
 * (l - x)^{exponent}, & x < l \\
 * 0,                  & l \leq x \leq r \\
 * (x - r)^{exponent}, & r < x
 * \right.
 * \f}
 * @param T Scalar data type.
 * @param l Left boundary of the piecewise function.
 * @param r Right boundary of the piecewise function.
 * @param exponent Power exponent used in the function.
 * @param x Input value for which f(x) is computed.
 * @return The computed value of f(x).
 */
template <typename T>
static __device__ T ComputePiecewiseFunction(const T l, const T r,
                                             const T exponent, const T x) {
  if (x < l) return std::pow(l - x, exponent);
  if (r < x) return std::pow(x - r, exponent);
  return 0;
}

template <typename T>
static __device__ T ComputePiecewiseFunctionGrad(const T l, const T r,
                                                 const T exponent, const T x) {
  if (x < l) return -exponent * std::pow(l - x, exponent - 1);
  if (r < x) return exponent * std::pow(x - r, exponent - 1);
  return 0;
}

/**
 * @brief CUDA kernel to compute the forward pass of the energy well function.
 */ 
template <typename T>
__global__ void ComputeEnergyWellForward(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const T *well_energy_function_exponent, int32_t *selected_crs,
    T *integral_output, int32_t *inst_cr_avail_map, int32_t *site2cr_map,
    int32_t *cr2slr_map, int32_t *inst2slr_map,
    uint8_t *inst_cnp_sll_optim_mask, int32_t x_layout_size,
    int32_t y_layout_size, int32_t num_crs, int32_t num_insts) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_insts) return;

  T       center_x = inst_pos[i << 1];
  T       center_y = inst_pos[i << 1 | 1];
  int32_t ix       = min(static_cast<int32_t>(center_x), x_layout_size - 1);
  int32_t iy       = min(static_cast<int32_t>(center_y), y_layout_size - 1);
  ix               = max(0, ix);
  iy               = max(0, iy);

  int32_t loc_cr_idx   = site2cr_map[iy + ix * y_layout_size];
  int32_t loc_slr_idx  = cr2slr_map[loc_cr_idx];
  bool    consider_sll = inst_cnp_sll_optim_mask[i];
  int32_t tgt_slr_idx  = consider_sll ? inst2slr_map[i] : -1;
  if (inst_cr_avail_map[i * num_crs + loc_cr_idx] &&
      (tgt_slr_idx == loc_slr_idx)) {
    selected_crs[i]    = loc_cr_idx;
    integral_output[i] = 0;
    return;
  }

  T       exponent            = well_energy_function_exponent[i];
  T       min_dist            = FLT_MAX;
  int32_t selected_cr         = -1;
  T       min_dist_withSll    = FLT_MAX;
  int32_t selected_cr_withSll = -1;

  for (int32_t cr_id = 0; cr_id < num_crs; cr_id++) {
    if (!inst_cr_avail_map[i * num_crs + cr_id]) continue;
    T box_xl = well_boxes[cr_id << 2];
    T box_yl = well_boxes[cr_id << 2 | 1];
    T box_xr = well_boxes[cr_id << 2 | 2];
    T box_yr = well_boxes[cr_id << 2 | 3];

    T temp_x = ComputePiecewiseFunction(box_xl, box_xr, exponent, center_x);
    T temp_y = ComputePiecewiseFunction(box_yl, box_yr, exponent, center_y);
    T temp   = temp_x + temp_y;

    if (temp < min_dist) {
      min_dist    = temp;
      selected_cr = cr_id;
    }

    if (consider_sll && cr2slr_map[cr_id] == tgt_slr_idx) {
      if (temp < min_dist_withSll) {
        min_dist_withSll    = temp;
        selected_cr_withSll = cr_id;
      }
    }
  }

  selected_crs[i] =
      (selected_cr_withSll != -1) ? selected_cr_withSll : selected_cr;
  integral_output[i] =
      (selected_cr_withSll != -1) ? min_dist_withSll : min_dist;
}

/**
 * @brief CUDA kernel to compute the backward pass of the energy well function.
 */ 
template <typename T>
__global__ void ComputeEnergyWellBackward(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const int32_t *selected_crs, const T *well_energy_function_exponent,
    const T *grad_output, T *grad_xy, int32_t num_insts) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_insts) return;
  int32_t cr_id    = selected_crs[i];
  T       exponent = well_energy_function_exponent[i];

  T box_xl = well_boxes[cr_id << 2];
  T box_yl = well_boxes[cr_id << 2 | 1];
  T box_xr = well_boxes[cr_id << 2 | 2];
  T box_yr = well_boxes[cr_id << 2 | 3];

  T center_x = inst_pos[i << 1];
  T center_y = inst_pos[i << 1 | 1];

  T temp_x = ComputePiecewiseFunctionGrad(box_xl, box_xr, exponent, center_x);
  T temp_y = ComputePiecewiseFunctionGrad(box_yl, box_yr, exponent, center_y);

  grad_xy[i << 1]     = grad_output[i] * temp_x;
  grad_xy[i << 1 | 1] = grad_output[i] * temp_y;
}

template <typename T>
void OPENPARF_NOINLINE ComputeEnergyWellForwardLauncher(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const T *well_energy_function_exponent, int32_t *selected_crs,
    T *integral_output, int32_t *inst_cr_avail_map, int32_t *site2cr_map,
    int32_t *cr2slr_map, int32_t *inst2slr_map,
    uint8_t *inst_cnp_sll_optim_mask, int32_t x_layout_size,
    int32_t y_layout_size, int32_t num_crs, int32_t num_insts,
    int32_t num_threads) {
  ComputeEnergyWellForward<<<ceilDiv(num_insts, 256), 256>>>(
      inst_pos, half_inst_sizes, well_boxes, well_energy_function_exponent,
      selected_crs, integral_output, inst_cr_avail_map, site2cr_map, cr2slr_map,
      inst2slr_map, inst_cnp_sll_optim_mask, x_layout_size, y_layout_size,
      num_crs, num_insts);
}

template <typename T>
void OPENPARF_NOINLINE ComputeEnergyWellBackwardLauncher(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const int32_t *selected_crs, const T *well_energy_function_exponent,
    const T *grad_output, T *grad_xy, int32_t num_insts, int32_t num_threads) {
  ComputeEnergyWellBackward<<<ceilDiv(num_insts, 256), 256>>>(
      inst_pos, half_inst_sizes, well_boxes, selected_crs,
      well_energy_function_exponent, grad_output, grad_xy, num_insts);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                         \
  template void ComputeEnergyWellForwardLauncher<T>(                        \
      const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,     \
      const T *well_energy_function_exponent, int32_t *selected_crs,        \
      T *integral_output, int32_t *inst_cr_avail_map, int32_t *site2cr_map, \
      int32_t *cr2slr_map, int32_t *inst2slr_map,                           \
      uint8_t *inst_cnp_sll_optim_mask, int32_t x_layout_size,              \
      int32_t y_layout_size, int32_t num_crs, int32_t num_insts,            \
      int32_t num_threads);                                                 \
  template void ComputeEnergyWellBackwardLauncher<T>(                       \
      const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,     \
      const int32_t *seleted_crs, const T *well_energy_function_exponent,   \
      const T *grad_output, T *grad_xy, int32_t num_insts,                  \
      int32_t num_threads);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE