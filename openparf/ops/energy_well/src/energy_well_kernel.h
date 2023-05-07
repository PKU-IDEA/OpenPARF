/**
 * File              : energy_well_kernel.cpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 09.17.2020
 * Last Modified Date: 09.17.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */

#include "util/util.h"
#include "database/clock_availability.h"
#include "database/placedb.h"

OPENPARF_BEGIN_NAMESPACE

using CoordinateType = database::PlaceDB::CoordinateType;

/**
 * @brief Compute the integral of piecewise power function f(x).
 * @details The piecewise power function is \f{equation*}{
 * f(x) = \left\{
 * (l - x)^{exponent}, & x < l \\
 * 0,                  & l \leq x \leq r \\
 * (x - r)^{exponent}, & r < x
 * \right.
 * \f}
 * @tparam T Scalar data type.
 * @param l
 * @param r
 * @param exponent
 * @param x
 * @return
 */
template<typename T, typename = typename std::enable_if<std::is_scalar<T>::value>::type>
inline static T ComputePiecewiseFunction(const T l, const T r, const T exponent, const T x) {
    if (x < l) return std::pow(l - x, exponent);
    if (r < x) return std::pow(x - r, exponent);
    return 0;
}

template<typename T, typename = typename std::enable_if<std::is_scalar<T>::value>::type>
inline static T ComputePiecewiseFunctionGrad(const T l, const T r, const T exponent, const T x) {
    if (x < l) return - exponent * std::pow(l - x, exponent - 1);
    if (r < x) return exponent * std::pow(x - r, exponent - 1);
    return 0;
}

template<typename T, typename = typename std::enable_if<std::is_scalar<T>::value>::type>
void ComputeEnergyWellForward(const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
                              const T *well_energy_function_exponent, int32_t *selected_crs,
                              T *integral_output,
                              int32_t *inst_cr_avail_map,
                              LayoutXy2GridIndexFunctorType<CoordinateType> xy_to_cr_func,
                              int32_t num_crs,
                              int32_t num_insts,
                              int32_t num_threads) {
    int32_t chunk_size = std::max(int32_t(num_insts / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int32_t i = 0; i < num_insts; i++) {
        T       center_x    = inst_pos[i << 1];
        T       center_y    = inst_pos[i << 1 | 1];
        int32_t loc_cr_idx = xy_to_cr_func(center_x, center_y);
        if(inst_cr_avail_map[i * num_crs + loc_cr_idx]){
            selected_crs[i] = loc_cr_idx;
            integral_output[i] = 0;
            continue;
        }
        T       exponent    = well_energy_function_exponent[i];
        T       min_dist    = std::numeric_limits<T>::max();
        int32_t selected_cr = -1;
        for(int32_t cr_id = 0; cr_id < num_crs; cr_id++){
            if(!inst_cr_avail_map[i * num_crs + cr_id]){
                continue;
            }
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
        }
        openparfAssert(selected_cr != -1);
        selected_crs[i]    = selected_cr;
        integral_output[i] = min_dist;
    }
}

template<typename T, typename = typename std::enable_if<std::is_scalar<T>::value>::type>
void ComputeEnergyWellBackward(const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
                               const int32_t *selected_crs, const T *well_energy_function_exponent,
                               const T *grad_output, T *grad_xy, int32_t num_insts,
                               int32_t num_threads) {
    int32_t chunk_size = std::max(int32_t(num_insts / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int32_t i = 0; i < num_insts; i++) {
        int32_t cr_id = selected_crs[i];
        T exponent = well_energy_function_exponent[i];

        T box_xl = well_boxes[cr_id << 2];
        T box_yl = well_boxes[cr_id << 2 | 1];
        T box_xr = well_boxes[cr_id << 2 | 2];
        T box_yr = well_boxes[cr_id << 2 | 3];

        T center_x = inst_pos[i << 1];
        T center_y = inst_pos[i << 1 | 1];

        /*
        T half_width  = half_inst_sizes[i << 1];
        T half_height = half_inst_sizes[i << 1 | 1];
        T xl          = center_x - half_width;
        T xh          = center_x + half_width;
        T yl          = center_y - half_height;
        T yh          = center_y + half_height;

        T temp_x = ComputePiecewiseFunction(box_xl, box_xr, exponent, xh) -
                   ComputePiecewiseFunction(box_xl, box_xr, exponent, xl);
        T temp_y = ComputePiecewiseFunction(box_yl, box_yr, exponent, yh) -
                   ComputePiecewiseFunction(box_yl, box_yr, exponent, yl);

        grad_xy[i << 1]     = grad_output[i] * (yh - yl) * temp_x;
        grad_xy[i << 1 | 1] = grad_output[i] * (xh - xl) * temp_y;
        */

        T temp_x = ComputePiecewiseFunctionGrad(box_xl, box_xr, exponent, center_x);
        T temp_y = ComputePiecewiseFunctionGrad(box_yl, box_yr, exponent, center_y);

//        if (std::isnan(grad_output[i])){
//            std::cerr << grad_output[i] << std::endl;
//        }
        openparfAssert(!std::isnan(grad_output[i]));

        grad_xy[i << 1]     = grad_output[i] * temp_x;
        grad_xy[i << 1 | 1] = grad_output[i] * temp_y;
    }
}

template<typename T>
void OPENPARF_NOINLINE
ComputeEnergyWellForwardLauncher(const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
                                 const T *well_energy_function_exponent, int32_t *selected_crs,
                                 T *integral_output,
                                 int32_t *inst_cr_avail_map,
                                 LayoutXy2GridIndexFunctorType<CoordinateType> xy_to_cr_func,
                                 int32_t num_crs, int32_t num_insts, int32_t num_threads) {
    ComputeEnergyWellForward(inst_pos, half_inst_sizes, well_boxes,
                             well_energy_function_exponent,
                             selected_crs, integral_output, inst_cr_avail_map, xy_to_cr_func,
                             num_crs, num_insts, num_threads);
}

template<typename T>
void OPENPARF_NOINLINE ComputeEnergyWellBackwardLauncher(
        const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
        const int32_t *selected_crs, const T *well_energy_function_exponent, const T *grad_output,
        T *grad_xy, int32_t num_insts, int32_t num_threads) {
    ComputeEnergyWellBackward(inst_pos, half_inst_sizes, well_boxes, selected_crs,
                              well_energy_function_exponent, grad_output, grad_xy, num_insts,
                              num_threads);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                                                                \
    template void ComputeEnergyWellForwardLauncher<T>(                                             \
            const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,                      \
            const T *well_energy_function_exponent, int32_t *selected_crs, T *integral_output,     \
            int32_t *                                     inst_cr_avail_map,                       \
            LayoutXy2GridIndexFunctorType<CoordinateType> xy_to_cr_func, int32_t num_crs,          \
            int32_t num_insts, int32_t num_threads);                                               \
    template void ComputeEnergyWellBackwardLauncher<T>(                                            \
            const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,                      \
            const int32_t *seleted_crs, const T *well_energy_function_exponent,                    \
            const T *grad_output, T *grad_xy, int32_t num_insts, int32_t num_threads);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

OPENPARF_END_NAMESPACE