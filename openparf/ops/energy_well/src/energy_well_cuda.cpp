/**
 * File              : energy_well.cpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 09.17.2020
 * Last Modified Date: 01.10.2024
 * Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>
 */

#include "database/placedb.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template <typename T>
void ComputeEnergyWellForwardLauncher(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const T *well_energy_function_exponent, int32_t *selected_crs,
    T *integral_output, int32_t *inst_cr_avail_map, int32_t *site2cr_map,
    int32_t *cr2slr_map, int32_t *inst2slr_map,
    uint8_t *inst_cnp_sll_optim_mask, int32_t x_layout_size,
    int32_t y_layout_size, int32_t num_crs, int32_t num_insts,
    int32_t num_threads);

template <typename T>
void ComputeEnergyWellBackwardLauncher(
    const T *inst_pos, const T *half_inst_sizes, const T *well_boxes,
    const int32_t *seleted_crs, const T *well_energy_function_exponent,
    const T *grad_output, T *grad_xy, int32_t num_insts, int32_t num_threads);

std::tuple<at::Tensor, at::Tensor> EnergyWellForward(
    at::Tensor inst_pos, at::Tensor half_inst_sizes, at::Tensor well_boxes,
    at::Tensor inst_cr_avail_map, at::Tensor energy_function_exponents,
    int32_t num_crs, database::PlaceDB &placedb, at::Tensor site2cr_map,
    at::Tensor cr2slr_map, at::Tensor inst2slr_map,
    at::Tensor inst_cnp_sll_optim_mask) {
  CHECK_FLAT_CUDA(inst_pos);
  CHECK_EVEN(inst_pos);
  CHECK_FLAT_CUDA(half_inst_sizes);
  CHECK_EVEN(half_inst_sizes);
  CHECK_FLAT_CUDA(well_boxes);
  CHECK_DIVISIBLE(well_boxes, 4);
  CHECK_FLAT_CUDA(energy_function_exponents);
  CHECK_FLAT_CUDA(inst_cr_avail_map);
  CHECK_CONTIGUOUS(inst_cr_avail_map);
  CHECK_FLAT_CUDA(site2cr_map);
  CHECK_FLAT_CUDA(cr2slr_map);
  CHECK_FLAT_CUDA(inst2slr_map);
  CHECK_CONTIGUOUS(inst2slr_map);
  CHECK_FLAT_CUDA(inst_cnp_sll_optim_mask);
  CHECK_CONTIGUOUS(inst_cnp_sll_optim_mask);

  int32_t num_insts     = inst_pos.numel() >> 1;
  int32_t x_layout_size = placedb.db()->layout().siteMap().width();
  int32_t y_layout_size = placedb.db()->layout().siteMap().height();

  at::Tensor integral_output = at::zeros({num_insts}, inst_pos.options());
  at::Tensor selected_crs =
      at::zeros({num_insts}, inst_pos.options()).to(torch::kInt32);

  OPENPARF_DISPATCH_FLOATING_TYPES(
      inst_pos, "ComputeEnergyWellForwardLauncher", [&] {
        ComputeEnergyWellForwardLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(inst_pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(half_inst_sizes, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(well_boxes, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(energy_function_exponents, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(selected_crs, int32_t),
            OPENPARF_TENSOR_DATA_PTR(integral_output, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(inst_cr_avail_map, int32_t),
            OPENPARF_TENSOR_DATA_PTR(site2cr_map, int32_t),
            OPENPARF_TENSOR_DATA_PTR(cr2slr_map, int32_t),
            OPENPARF_TENSOR_DATA_PTR(inst2slr_map, int32_t),
            OPENPARF_TENSOR_DATA_PTR(inst_cnp_sll_optim_mask, uint8_t),
            x_layout_size, y_layout_size, num_crs, num_insts,
            at::get_num_threads());
      });
  return {integral_output, selected_crs};
}

at::Tensor EnergyWellBackward(at::Tensor inst_pos, at::Tensor half_inst_sizes,
                              at::Tensor well_boxes, at::Tensor selected_crs,
                              at::Tensor energy_function_exponents,
                              at::Tensor grad_output) {
  CHECK_FLAT_CUDA(inst_pos);
  CHECK_EVEN(inst_pos);
  CHECK_FLAT_CUDA(half_inst_sizes);
  CHECK_EVEN(half_inst_sizes);
  CHECK_FLAT_CUDA(well_boxes);
  CHECK_DIVISIBLE(well_boxes, 4);
  CHECK_FLAT_CUDA(selected_crs);
  CHECK_FLAT_CUDA(energy_function_exponents);
  CHECK_FLAT_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);

  int32_t num_insts = inst_pos.numel() >> 1;

  at::Tensor grad_xy = at::zeros({num_insts, 2}, inst_pos.options());
  openparfAssert(OPENPARF_TENSOR_SCALARTYPE(inst_pos) ==
                 OPENPARF_TENSOR_SCALARTYPE(grad_output));
  //    std::cerr << "grad_output" << std::endl;
  //    std::cerr << grad_output << std::endl;
  //    auto grad_output_a = grad_output.accessor<double, 1>();
  //    for(int i = 0; i < grad_output.size(0); i++){
  //        if(std::isnan(grad_output_a[i])){
  //            std::cerr << "nan on " << i << std::endl;
  //            exit(-1);
  //        }
  //    }

  OPENPARF_DISPATCH_FLOATING_TYPES(
      inst_pos, "ComputeEnergyWellBackwardLauncher", [&] {
        ComputeEnergyWellBackwardLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(inst_pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(half_inst_sizes, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(well_boxes, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(selected_crs, int32_t),
            OPENPARF_TENSOR_DATA_PTR(energy_function_exponents, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(grad_output, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(grad_xy, scalar_t), num_insts,
            at::get_num_threads());
      });

  return grad_xy;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::EnergyWellForward,
        "Energy Well Forward(CUDA)");
  m.def("backward", &OPENPARF_NAMESPACE::EnergyWellBackward,
        "Energy Well Backward(CUDA)");
}
