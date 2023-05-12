/**
 * File              : energy_well.cpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 09.17.2020
 * Last Modified Date: 09.17.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */

#include "util/torch.h"
#include "util/util.h"

#include "energy_well_kernel.h"
#include "database/clock_availability.h"
#include "database/placedb.h"


OPENPARF_BEGIN_NAMESPACE

using CoordinateType = database::PlaceDB::CoordinateType;

std::tuple<at::Tensor, at::Tensor>
EnergyWellForward(at::Tensor inst_pos, at::Tensor half_inst_sizes, at::Tensor well_boxes,
                  at::Tensor inst_cr_avail_map,
                  at::Tensor energy_function_exponents,
                  int32_t num_crs,
                  database::PlaceDB &placedb) {
    CHECK_FLAT_CPU(inst_pos);
    CHECK_EVEN(inst_pos);
    CHECK_FLAT_CPU(half_inst_sizes);
    CHECK_EVEN(half_inst_sizes);
    CHECK_FLAT_CPU(well_boxes);
    CHECK_DIVISIBLE(well_boxes, 4);
    CHECK_FLAT_CPU(energy_function_exponents);
    CHECK_FLAT_CPU(inst_cr_avail_map);
    CHECK_CONTIGUOUS(inst_cr_avail_map);

    int32_t num_insts = inst_pos.numel() >> 1;

    namespace arg = std::placeholders;

    LayoutXy2GridIndexFunctorType<CoordinateType> xy_to_cr_func =
        std::bind(&database::PlaceDB::XyToCrIndex, &placedb, arg::_1, arg::_2);

    at::Tensor integral_output = at::zeros({num_insts}, inst_pos.options());
    at::Tensor selected_crs    = at::zeros({num_insts}, inst_pos.options()).to(torch::kInt32);

    OPENPARF_DISPATCH_FLOATING_TYPES(inst_pos, "ComputeEnergyWellForwardLauncher", [&] {
        ComputeEnergyWellForwardLauncher<scalar_t>(
                OPENPARF_TENSOR_DATA_PTR(inst_pos, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(half_inst_sizes, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(well_boxes, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(energy_function_exponents, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(selected_crs, int32_t),
                OPENPARF_TENSOR_DATA_PTR(integral_output, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(inst_cr_avail_map, int32_t),
                xy_to_cr_func,
                num_crs,
                num_insts,
                at::get_num_threads());
    });
    return {integral_output, selected_crs};
}

at::Tensor EnergyWellBackward(at::Tensor inst_pos, at::Tensor half_inst_sizes,
                              at::Tensor well_boxes, at::Tensor selected_crs,
                              at::Tensor energy_function_exponents, at::Tensor grad_output) {
    CHECK_FLAT_CPU(inst_pos);
    CHECK_EVEN(inst_pos);
    CHECK_FLAT_CPU(half_inst_sizes);
    CHECK_EVEN(half_inst_sizes);
    CHECK_FLAT_CPU(well_boxes);
    CHECK_DIVISIBLE(well_boxes, 4);
    CHECK_FLAT_CPU(selected_crs);
    CHECK_FLAT_CPU(energy_function_exponents);
    CHECK_FLAT_CPU(grad_output);
    CHECK_CONTIGUOUS(grad_output);

    int32_t num_insts = inst_pos.numel() >> 1;

    at::Tensor grad_xy = at::zeros({num_insts, 2}, inst_pos.options());
    openparfAssert(OPENPARF_TENSOR_SCALARTYPE(inst_pos) == OPENPARF_TENSOR_SCALARTYPE(grad_output));
//    std::cerr << "grad_output" << std::endl;
//    std::cerr << grad_output << std::endl;
//    auto grad_output_a = grad_output.accessor<double, 1>();
//    for(int i = 0; i < grad_output.size(0); i++){
//        if(std::isnan(grad_output_a[i])){
//            std::cerr << "nan on " << i << std::endl;
//            exit(-1);
//        }
//    }

    OPENPARF_DISPATCH_FLOATING_TYPES(inst_pos, "ComputeEnergyWellBackwardLauncher", [&] {
        ComputeEnergyWellBackwardLauncher<scalar_t>(
                OPENPARF_TENSOR_DATA_PTR(inst_pos, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(half_inst_sizes, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(well_boxes, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(selected_crs, int32_t),
                OPENPARF_TENSOR_DATA_PTR(energy_function_exponents, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(grad_output, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(grad_xy, scalar_t), num_insts, at::get_num_threads());
    });

    return grad_xy;
}

at::Tensor genSite2CrMap(database::PlaceDB &placedb) {
    auto options = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false);
    int32_t    x_layout_size = placedb.db()->layout().siteMap().width();
    int32_t    y_layout_size = placedb.db()->layout().siteMap().height();
    at::Tensor site2cr_map   = at::zeros({x_layout_size, y_layout_size}, options);
    auto       site2cr_map_a = site2cr_map.accessor<int32_t, 2>();
    for (int32_t ix = 0; ix < x_layout_size; ix++) {
        for (int32_t iy = 0; iy < y_layout_size; iy++) {
            site2cr_map_a[ix][iy] = placedb.XyToCrIndex(ix, iy);
        }
    }
    return site2cr_map;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::EnergyWellForward, "Energy Well Forward");
    m.def("backward", &OPENPARF_NAMESPACE::EnergyWellBackward, "Energy Well Backward");
    m.def("genSite2CrMap", &OPENPARF_NAMESPACE::genSite2CrMap,
          "Generate Site to Clock Region Mapping");
}
