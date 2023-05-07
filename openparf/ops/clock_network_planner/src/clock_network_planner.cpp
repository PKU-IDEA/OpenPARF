/**
 * @file   clock_network_planner.cpp
 * @author Yibai Meng
 * @date   Sep 2020
 */
#include <iostream>
#include <tuple>

#include "database/placedb.h"
#include "util/torch.h"

#include "ops/clock_network_planner/src/clock_network_planner_kernel.cpp"

OPENPARF_BEGIN_NAMESPACE

// @brief Wrapper to deal with OPENPARF_DISPATCH_FLOATING_TYPES
struct Dummy {
    Dummy() {}
    void set(at::ScalarType s) {
        _scalar_type = s;
    }
    // for backward compatibility when PyTorch API changes
    at::ScalarType scalar_type() const {
        return _scalar_type;
    }
    // for backward compatibility when PyTorch API changes
    at::ScalarType scalarType() const {
        return scalar_type();
    }
    at::ScalarType _scalar_type;
};

// Do we reallly need such a wrapper?
// @brief Initialize the needed data structures
// @param[in] placedb  placement database
// @param[in] float_type float64 or float32, the datatype the position vector uses
void initializeClockNetworkPlanner(database::PlaceDB const &placedb, std::string float_type) {
    openparfAssert(float_type == "float64" or float_type == "float32");
    Dummy d;
    if (float_type == "float64") {
        d.set(at::ScalarType::Double);
    } else if (float_type == "float32") {
        d.set(at::ScalarType::Float);
    }
    OPENPARF_DISPATCH_FLOATING_TYPES(d, "cnpInitLauncher", [&] {
        cnpInitLauncher<scalar_t>(placedb);
    });
}

// @brief Assign clock region based on the position of the instances
// @param[in] pos Current position of the instances, #(num_insts, 2) shaped torch.Tensor
// @param[in] pyparam Python object recording the other parameters
// @param[in] placedb placement database
// @return 1st tensor: nodeToCRclock, a #(num_insts) torch.Tensor. res[i] is the clock region if to
// which instance i is assigned.
//         2nd tensor: clkAvailCR, a #(num_clk_nets, num_clk_region) shaped torch.Tensor. If
//         res[clk_idx][cr_id] is 1, then clock clk_id is allowed in clock region cr_id, otherwise
//         clk_idx is not allowed there.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::vector<int32_t>>>
runClockRegionAssignment(database::PlaceDB const &placedb,
                         py::object    pyparam,   // python object recording the other parameters
                         torch::Tensor pos, torch::Tensor inflated_areas) {
    CHECK_FLAT_CPU(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT_CPU(inflated_areas);
    CHECK_CONTIGUOUS(inflated_areas);

    size_t num_insts = pos.numel() >> 1;
    openparfAssert(placedb.numInsts() == num_insts);

    utplacefx::Parameters::archClockRegionClockCapacity = pyparam.attr("maximum_clock_per_clock_region").cast<decltype(utplacefx::Parameters::archClockRegionClockCapacity)>();

    auto const cr_map       = placedb.db()->layout().clockRegionMap();
    auto const numClockNets = placedb.numClockNets();
    auto       options      = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false);
    auto nodeToCR = torch::full({placedb.numInsts()}, InvalidIndex<int32_t>::value, options);
    auto options2 = torch::TensorOptions()
                            .dtype(torch::kUInt8)
                            .layout(torch::kStrided)
                            .device(torch::kCPU)
                            .requires_grad(false);
    auto clkAvailCR = torch::full({numClockNets, cr_map.width() * cr_map.height()},
                                  InvalidIndex<uint8_t>::value, options2);
    OPENPARF_DISPATCH_FLOATING_TYPES(pos, "clockNetworkPlannerLauncher", [&] {
        clockNetworkPlannerLauncher<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                              torch::get_num_threads(),
                                              OPENPARF_TENSOR_DATA_PTR(nodeToCR, int32_t),
                                              OPENPARF_TENSOR_DATA_PTR(clkAvailCR, uint8_t),
                                              OPENPARF_TENSOR_DATA_PTR(inflated_areas, scalar_t));
    });
//    std::ofstream of("ck_avail_cr.txt",  std::ofstream::out);
    auto clkAvailCRAccessor = clkAvailCR.accessor<uint8_t, 2>();
    openparfPrint(kDebug, "Clock Region Available Clock Region:\n");
    for (int32_t ck_id = 0; ck_id < numClockNets; ck_id++) {
        openparfPrint(kDebug, "============ ck_id: %d =============\n", ck_id);
        for (int32_t x_cr_id = 0; x_cr_id < cr_map.width(); x_cr_id++) {
            std::stringstream ss;
            for (int32_t y_cr_id = 0; y_cr_id < cr_map.height(); y_cr_id++) {
                int32_t cr_id = y_cr_id + x_cr_id * cr_map.height();
                ss << int32_t(clkAvailCRAccessor[ck_id][cr_id]) << " ";
            }
            openparfPrint(kDebug, "%s\n", ss.str().c_str());
//            of << ss.str().c_str() << std::endl;
        }
    }
//    of.close();
    size_t num_crs  = cr_map.width() * cr_map.height();
    using IndexType = database::PlaceDB::IndexType;
    at::Tensor instCrAvailMap =
            torch::zeros({static_cast<long>(num_insts), static_cast<long>(num_crs)}, options);
    auto instCrAvailMapAccessor = instCrAvailMap.accessor<int32_t, 2>();
    std::vector<std::vector<int32_t>>   instAvailCR;
    std::vector<std::vector<IndexType>> instToClocks = placedb.instToClocks();
    for (size_t i = 0; i < num_insts; i++) {
        const std::vector<IndexType> &cks = instToClocks[i];
        std::vector<int32_t>          availCR;
        if (cks.empty()) {
            // this instance is not connected to any clock net, so it can be assigned to any clock
            // regions.
            for(size_t j = 0; j < num_crs; j++){
                instCrAvailMapAccessor[i][j] = 1;
                availCR.emplace_back(j);
            }
            instAvailCR.emplace_back(availCR);
            continue;
        }
        at::Tensor availMap = torch::ones({static_cast<long>(num_crs)}, options2);
        uint8_t *crs_begin = OPENPARF_TENSOR_DATA_PTR(availMap, uint8_t);
        for(auto &ck_id : cks){
            // TODO(Jing Mai, jingmai@pku.edu.cn): `torch 1.0.0` don't support `at::bitwise_and`.
//            crs = at::bitwise_and(crs, clkAvailCR.index({ck_id, "..."}));
            uint8_t *temp = OPENPARF_TENSOR_DATA_PTR(clkAvailCR, uint8_t) + num_crs * ck_id;
            for(size_t j = 0; j < num_crs; j++){
                crs_begin[j] = crs_begin[j] & temp[j];
            }
        }
        for(size_t j = 0; j < num_crs; j++){
            if(crs_begin[j] == 1){
                instCrAvailMapAccessor[i][j] = 1;
                availCR.emplace_back(j);
            }
        }
        openparfAssert(!availCR.empty());
        instAvailCR.emplace_back(availCR);
    }
    return {nodeToCR, clkAvailCR, instCrAvailMap, instAvailCR};
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::runClockRegionAssignment,
          "Clock network planner forward");
    m.def("init", &OPENPARF_NAMESPACE::initializeClockNetworkPlanner,
          "Intialize clock network planner");
}
