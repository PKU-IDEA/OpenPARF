/**
 * File              : clock_demand_estimation_factory.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 04.16.2021
 * Last Modified Date: 04.16.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "clock_demand_estimation_factory.h"
#include "clock_network_planner.h"
#include "util/message.h"
#include <iomanip>
#include <sstream>

namespace clock_network_planner {

ClockDemandEstimation
ClockDemandEstimationFactory::estimateClockDemand(ClockNetworkPlanner cnp,
                                                  CrId2CkSinkSet      cr_id2ck_sink_set) {
    using IndexType      = ClockDemandEstimationImpl::IndexType;
    auto  ck_estimation  = detail::makeClockDemandEstimation();
    auto &ck2cr_box_map  = ck_estimation.ck2cr_box_map();
    auto &ck_demand_grid = ck_estimation.ck_demand_grid();
    ck2cr_box_map.clear();
    IndexType x_cr_num = cnp.x_crs_num();
    IndexType y_cr_num = cnp.y_crs_num();
    for (IndexType x_cr_id = 0; x_cr_id < x_cr_num; x_cr_id++) {
        for (int y_cr_id = 0; y_cr_id < y_cr_num; ++y_cr_id) {
            IndexType cr_id = x_cr_id * y_cr_num + y_cr_id;
            for (IndexType ck_id : cr_id2ck_sink_set.at(cr_id)) {
                if (ck2cr_box_map.find(ck_id) == ck2cr_box_map.end()) {
                    ck2cr_box_map[ck_id].set(x_cr_id, y_cr_id, x_cr_id, y_cr_id);
                } else {
                    ck2cr_box_map[ck_id].join(Box<IndexType>(x_cr_id, y_cr_id, x_cr_id, y_cr_id));
                }
            }
        }
    }
    ck_demand_grid.clear();
    ck_demand_grid.resize(x_cr_num, y_cr_num);
    for (auto &p : ck2cr_box_map) {
        IndexType ck_id = p.first;
        auto &    box   = p.second;
        for (IndexType i = box.xl(); i <= box.xh(); i++) {
            for (IndexType j = box.yl(); j <= box.yh(); ++j) {
                ck_demand_grid.at(i, j).insert(ck_id);
            }
        }
    }
    openparfPrint(kDebug, "Clock Demand Grid:\n");
    for (int x_cr_id = 0; x_cr_id < x_cr_num; x_cr_id++) {
        std::stringstream ss;
        for (int y_cr_id = 0; y_cr_id < y_cr_num; y_cr_id++) {
            ss << std::setfill(' ') << std::setw(2) << ck_demand_grid.at(x_cr_id, y_cr_id).size()
               << " ";
        }
        openparfPrint(kDebug, "%s\n", std::string(ss.str()).c_str());
    }
//    for (auto &p : ck2cr_box_map) {
//        IndexType ck_id = p.first;
//        auto &    box   = p.second;
//        openparfPrint(kDebug, "%d: xl = %d, yl = %d, xh = %d, yh = %d\n", ck_id, box.xl(), box.yl(),
//                      box.xh(), box.yh());
//    }
    return ck_estimation;
}
}   // namespace clock_network_planner
