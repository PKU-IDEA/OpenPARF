/**
 * File              : utplace2_cnp_torch.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 04.20.2021
 * Last Modified Date: 04.20.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "database/placedb.h"
#include "ops/utplace2_cnp/src/clock_region_assignment_result_factory.h"
#include "util/torch.h"
#include "util/message.h"

#include "ops/utplace2_cnp/src/clock_network_planner_factory.h"

OPENPARF_BEGIN_NAMESPACE

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::vector<int32_t>>>
runClockRegionAssignment(database::PlaceDB &placedb, torch::Tensor pos,
                         torch::Tensor inflated_areas) {
    CHECK_FLAT_CPU(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT_CPU(inflated_areas);
    CHECK_CONTIGUOUS(inflated_areas);

    size_t num_insts = pos.numel() >> 1;
    openparfAssert(placedb.numInsts() == num_insts);

    using namespace clock_network_planner;

    int32_t     num_cks  = placedb.numClockNets();
    auto const &cr_map   = placedb.db()->layout().clockRegionMap();
    int32_t     x_cr_num = cr_map.width();
    int32_t     y_cr_num = cr_map.height();
    int32_t     num_crs  = x_cr_num * y_cr_num;
    auto        options  = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false);
    auto options2 = torch::TensorOptions()
                            .dtype(torch::kUInt8)
                            .layout(torch::kStrided)
                            .device(torch::kCPU)
                            .requires_grad(false);

    auto cnp = ClockNetworkPlannerFactory::genClockNetworkPlannerFromPlaceDB(placedb);

    openparfPrint(kDebug, "cnp utplace2 enable packing: %d\n", cnp.enable_packing());
    openparfPrint(kDebug, "clock region capacity: %d\n", cnp.clock_region_capacity());
    openparfPrint(kDebug, "cnp utplace2 maxinum pruning per iteration: %d\n", cnp.max_num_clock_pruning_per_iteration());

    ClockNetworkPlanner::ClockAvailCRArray ck_avail_crs;
    OPENPARF_DISPATCH_FLOATING_TYPES(pos, "clockNetworkPlannerLauncher", [&] {
        cnp.run<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                          OPENPARF_TENSOR_DATA_PTR(inflated_areas, scalar_t), ck_avail_crs);
    });
    openparfAssert(ck_avail_crs.size() == num_cks);

    auto clk_avail_cr        = torch::full({num_cks, num_crs}, 0, options2);
    auto clk_avail_cr_a      = clk_avail_cr.accessor<uint8_t, 2>();
    auto inst_cr_avail_map   = torch::full({static_cast<long>(num_insts), num_crs}, 1, options);
    auto inst_cr_avail_map_a = inst_cr_avail_map.accessor<int32_t, 2>();
    auto inst_to_nearest_avail_cr   = torch::full({static_cast<long>(num_insts)}, 0, options);
    auto inst_to_nearest_avail_cr_a = inst_to_nearest_avail_cr.accessor<int32_t, 1>();
    std::vector<std::vector<int32_t>> inst_avail_crs(num_insts);

//    std::ifstream ifs("ck_avail_cr.txt", std::ifstream::in);
//    for(int32_t ck_id = 0; ck_id <num_cks; ck_id++){
//        for(int cr_id = 0; cr_id < num_crs; cr_id++){
//            ifs >> ck_avail_crs[ck_id][cr_id];
//        }
//    }
//    ifs.close();

    for (int32_t ck_id = 0; ck_id < num_cks; ck_id++) {
        openparfAssert(ck_avail_crs[ck_id].size() == num_crs);
        for (int32_t cr_id = 0; cr_id < num_crs; cr_id++) {
            clk_avail_cr_a[ck_id][cr_id] = ck_avail_crs[ck_id][cr_id];
        }
    }
    openparfPrint(kDebug, "Clock Region Available Clock Region:\n");
    for (int32_t ck_id = 0; ck_id < num_cks; ck_id++) {
        openparfPrint(kDebug, "============ ck_id: %d =============\n", ck_id);
        for (int32_t x_cr_id = 0; x_cr_id < x_cr_num; x_cr_id++) {
            std::stringstream ss;
            for (int32_t y_cr_id = 0; y_cr_id < y_cr_num; y_cr_id++) {
                int32_t cr_id = y_cr_id + x_cr_id * y_cr_num;
                ss << int32_t(clk_avail_cr_a[ck_id][cr_id]) << " ";
            }
            openparfPrint(kDebug, "%s\n", ss.str().c_str());
        }
    }
    // FIXME(jingmai@pku.edu.cn): generate the node to clock region by mcf instead of moving
    //  the nearest clock regions.
    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
        for (const PlaceDB::IndexType &ck_id : placedb.instToClocks()[inst_id]) {
            for (int i = 0; i < num_crs; i++) {
                if (!ck_avail_crs[ck_id][i]) { inst_cr_avail_map_a[inst_id][i] = 0; }
            }
        }
        using namespace torch::indexing;
        openparfAssert(inst_cr_avail_map.index({inst_id, "..."}).sum().item<int>() > 0);
        int32_t nearest_cr_id = -1;
        double  shortest_dist;
        for (int i = 0; i < num_crs; i++) {
            if (inst_cr_avail_map_a[inst_id][i]) {
                inst_avail_crs[inst_id].push_back(i);
                double dist = ClockRegionAssignmentResultFactory::getInstToCrDist(
                        cnp, inst_id, cr_map.at(i), placedb);
                if (nearest_cr_id == -1 || dist < shortest_dist) {
                    nearest_cr_id = i;
                    shortest_dist = dist;
                }
            }
        }
        openparfAssert(nearest_cr_id != -1);
        inst_to_nearest_avail_cr_a[inst_id] = nearest_cr_id;
    }
    return {inst_to_nearest_avail_cr, clk_avail_cr, inst_cr_avail_map, inst_avail_crs};
}
OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::runClockRegionAssignment,
          "Clock Network Planner(UTPlace 2.0)");
}
