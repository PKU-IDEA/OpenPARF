/**
 * File              : mcf_lg.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.10.2020
 * Last Modified Date: 10.20.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */
#include <cstdint>
#include <vector>

#include "util/message.h"
#include "util/torch.h"

#include "ops/mcf_lg/src/mcf_lg_kernel.h"

OPENPARF_BEGIN_NAMESPACE
template<typename T>
void MinCostFlowLegalizer<T>::add_sssir_instances(at::Tensor inst_ids, at::Tensor inst_weights,
                                                  at::Tensor inst_compatiable_sites) {
    // Assert correct shape of tensors
    CHECK_FLAT_CPU(inst_ids);
    CHECK_CONTIGUOUS(inst_ids);
    CHECK_FLAT_CPU(inst_weights);
    CHECK_CONTIGUOUS(inst_weights);
    CHECK_FLAT_CPU(inst_compatiable_sites);
    CHECK_DIVISIBLE(inst_compatiable_sites, 4);

    sssir_model_infos.emplace_back();
    auto &m = sssir_model_infos.back();
    m.hc_arcs.resize(num_half_column_regions);
    auto inst_ids_size = inst_ids.sizes();
    openparfAssertMsg(inst_ids_size.size() == 1, "inst_ids tensor must be 1D\n");
    m.num_insts                      = inst_ids_size.front();
    auto inst_compatiable_sites_size = inst_compatiable_sites.sizes();
    openparfAssertMsg(inst_compatiable_sites_size.size() == 2,
                      "inst_compatiable_sites tensor must be 2D\n");
    m.num_sites = inst_compatiable_sites_size.front();
    openparfAssertMsg(inst_compatiable_sites_size[1] == 4,
                      "inst_compatiable_sites must have a dimension of (#num_insts, 4)\n");
    // We do not copy to save time
    // Must check the validity of data before using them
    m.inst_ids               = inst_ids;
    m.inst_weights           = inst_weights;
    m.inst_compatiable_sites = inst_compatiable_sites;
    openparfPrint(MessageType::kDebug, "Added %i SSSIR instances and %i target sites\n",
                  m.num_insts, m.num_sites);
    for (int32_t i = 0; i < m.num_insts; i++) {
        m.instance_nodes.emplace_back(mcf_problem.graph.addNode());
        m.instance_arcs.emplace_back(
                mcf_problem.graph.addArc(mcf_problem.source, m.instance_nodes.back()));
        auto &arc                             = m.instance_arcs.back();
        mcf_problem.cost[arc]                 = 0;
        mcf_problem.capacity_lower_bound[arc] = 0;
        mcf_problem.capacity_upper_bound[arc] = 1;
    }
    for (int32_t i = 0; i < m.num_sites; i++) {
        m.site_nodes.emplace_back(mcf_problem.graph.addNode());
        m.site_arcs.emplace_back(mcf_problem.graph.addArc(m.site_nodes.back(), mcf_problem.drain));
        auto &arc                             = m.site_arcs.back();
        mcf_problem.cost[arc]                 = 0;
        mcf_problem.capacity_lower_bound[arc] = 0;
        mcf_problem.capacity_upper_bound[arc] = 1;
    }
}

template<typename T>
at::Tensor MinCostFlowLegalizer<T>::forward(at::Tensor pos) {
    if (honor_clock_region_constraints){
        openparfPrint(kDebug, "Clock constraints is activated for SSSIR legalization.\n");
    } else {
        openparfPrint(kDebug, "Clock constraints is NOT activated for SSSIR legalization.\n");
    }
    // https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    at::Tensor res = pos.clone().detach();
    // https://stackoverflow.com/questions/4942703/why-do-i-get-an-error-trying-to-call-a-template-member-function-with-an-explicit
    auto    pos_acc    = pos.template accessor<T, 2>();   //
    int32_t num_models = sssir_model_infos.size();
    openparfPrint(MessageType::kDebug, "Starting min cost flow legalization of %i SSSIR models\n",
                  num_models);

    calculate_max_distance(pos);
    for (int idx = 0; idx < num_models; idx++) {
        openparfPrint(kDebug, "Model %i:  %i instances, %i sites\n", idx,
                      sssir_model_infos[idx].num_insts, sssir_model_infos[idx].num_sites);
    }
    if(honor_clock_region_constraints) init_clock_constraints();
    bool    stop_flag   = false;
    bool    mcf_legal   = false;
    bool    clock_legal = false;
    int32_t iter        = 0;
    while (not stop_flag) {
        // purge_arcs
        int32_t total_num_insts = 0;
        for (int idx = 0; idx < num_models; idx++) {
            auto &m = sssir_model_infos[idx];
            total_num_insts += m.num_insts;
            T min_dis = m.dist_incr_per_iter * iter;
            T max_dis = m.dist_incr_per_iter * (iter + 1);
            openparfPrint(MessageType::kDebug,
                          "Iter %i model %i: check sites with distance smaller than %f\n", iter,
                          idx, max_dis);
            auto inst_id_acc                = m.inst_ids.template accessor<int32_t, 1>();
            auto inst_weight_acc            = m.inst_weights.template accessor<T, 1>();
            auto inst_compatiable_sites_acc = m.inst_compatiable_sites.template accessor<T, 2>();
            for (int32_t i = 0; i < m.num_insts; i++) {
                int32_t inst_id     = inst_id_acc[i];
                T       inst_weight = inst_weight_acc[i];
                T       x           = pos_acc[inst_id][0];
                T       y           = pos_acc[inst_id][1];
                IndexVectorType inst_slr_index;
                if (slr_aware_flag) {
                    inst_slr_index = xy_to_slr_functor(static_cast<int32_t>(x), static_cast<int32_t>(y));
                }
                for (int32_t s = 0; s < m.num_sites; s++) {
                    T       xl     = inst_compatiable_sites_acc[s][0];
                    T       xh     = inst_compatiable_sites_acc[s][2];
                    T       yl     = inst_compatiable_sites_acc[s][1];
                    T       yh     = inst_compatiable_sites_acc[s][3];
                    T       site_x = (xl + xh) / 2.0;
                    T       site_y = (yl + yh) / 2.0;
                    int32_t hc_id;
                    if(honor_clock_region_constraints)  {
                        hc_id = xy_to_half_column_functor((int32_t) site_x, (int32_t) site_y);
                    }
                    T       m_dist = std::abs(site_x - x) + std::abs(site_y - y);
                    if (m_dist >= min_dis and m_dist < max_dis and
                        (not honor_clock_region_constraints or
                         inst_clock_legal_at_site(inst_id, xl, yl))) {
                      if (slr_aware_flag) {
                        IndexVectorType site_slr_index = xy_to_slr_functor(static_cast<int32_t>(x), static_cast<int32_t>(y));
                        m_dist += (std::abs(site_slr_index[0] - inst_slr_index[0]) +
                                   5 * std::abs(site_slr_index[1] - inst_slr_index[1])) *
                                  m.max_distance;
                      }
                      m.inst_to_site_arcs.emplace_back(
                          inst_id, s,
                          mcf_problem.graph.addArc(m.instance_nodes[i], m.site_nodes[s]));
                      auto &arc                             = std::get<2>(m.inst_to_site_arcs.back());
                      mcf_problem.cost[arc]                 = (CostType)(m_dist * inst_weight * scale_factor);
                      mcf_problem.capacity_lower_bound[arc] = 0;
                      mcf_problem.capacity_upper_bound[arc] = 1;
                      if (honor_clock_region_constraints)
                        m.hc_arcs.at(hc_id).push_back(m.inst_to_site_arcs.size() - 1);
                    }
                }
            }
        }

        // Solve the mcf_problem
        mcf_problem.supply = total_num_insts;
        openparfAssert(mcf_problem.solve());

        // Now check if the results are legal
        stop_flag   = true;
        mcf_legal   = true;
        clock_legal = true;
        for (auto &m : sssir_model_infos) {
            int32_t total_flow = 0;
            for (auto const &arc : m.site_arcs) { total_flow += mcf_problem.solver.flow(arc); }
            if (total_flow != m.num_insts) {
                openparfPrint(MessageType::kDebug,
                              "Total flow %i not equal to num of inst %i for model %i\n",
                              total_flow, m.num_insts, m.model_id);
                stop_flag = false;
                mcf_legal = false;
            }
        }
        if (stop_flag) {
            // retrieve the assignment
            for (auto &m : sssir_model_infos) {
                auto inst_id_acc     = m.inst_ids.template accessor<int32_t, 1>();
                auto inst_weight_acc = m.inst_weights.template accessor<T, 1>();
                auto inst_compatiable_sites_acc =
                        m.inst_compatiable_sites.template accessor<T, 2>();
                for (auto const &arc : m.inst_to_site_arcs) {
                    if (mcf_problem.solver.flow(std::get<2>(arc))) {
                        int32_t inst_id = std::get<0>(arc);
                        int32_t s       = std::get<1>(arc);
                        T       xl      = inst_compatiable_sites_acc[s][0];
                        T       xh      = inst_compatiable_sites_acc[s][2];
                        T       yl      = inst_compatiable_sites_acc[s][1];
                        T       yh      = inst_compatiable_sites_acc[s][3];
                        T       site_x  = (xl + xh) / 2.0;
                        T       site_y  = (yl + yh) / 2.0;
                        res[inst_id][0] = site_x;
                        res[inst_id][1] = site_y;
                    }
                }
            }

            if (honor_clock_region_constraints) {
                calculate_half_column_scoreboard_cost(res);
                auto illegal_clocks = check_half_column_sat_and_prune_clock();
                openparfPrint(kDebug, "Iter %i: %i illegal clk-hc pairs\n", iter,
                              illegal_clocks.size());
                int32_t rm_edges = 0;
                if (not illegal_clocks.empty()) {
                    stop_flag   = false;
                    clock_legal = false;
                    for (auto &m : sssir_model_infos) {
                        for (auto clk_hc : illegal_clocks) {
                            int32_t illegal_clk = clk_hc.first;
                            int32_t hc_id       = clk_hc.second;
                            for (int32_t arc_id : m.hc_arcs.at(hc_id)) {
                                auto const &arc_tuple = m.inst_to_site_arcs[arc_id];
                                int32_t     inst_id   = std::get<0>(arc_tuple);
                                int32_t     s         = std::get<1>(arc_tuple);
                                auto &      arc       = std::get<2>(arc_tuple);
                                auto        inst_compatiable_sites_acc =
                                        m.inst_compatiable_sites.template accessor<T, 2>();
                                T xl     = inst_compatiable_sites_acc[s][0];
                                T xh     = inst_compatiable_sites_acc[s][2];
                                T yl     = inst_compatiable_sites_acc[s][1];
                                T yh     = inst_compatiable_sites_acc[s][3];
                                T site_x = (xl + xh) / 2.0;
                                T site_y = (yl + yh) / 2.0;
                                for (auto clk : inst_to_clock_indexes.at(inst_id)) {
                                    if (clk == illegal_clk) {
                                        if (mcf_problem.capacity_upper_bound[arc] != 0) rm_edges++;
                                        mcf_problem.capacity_upper_bound[arc] = 0;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                if (rm_edges != 0) openparfPrint(kDebug, "Removed %i illegal arcs\n", rm_edges);
            }
        }
        iter++;
    }
    return res;
}


template<typename T>
bool MinCostFlowLegalizer<T>::inst_clock_legal_at_site(int32_t inst_id, int32_t site_x,
                                                       int32_t site_y) {
    auto        cr_id = xy_to_clock_region_functor(site_x, site_y);
    auto        hc_id = xy_to_half_column_functor(site_x, site_y);
    auto const &sb    = hcsb_array.at(hc_id);
    for (auto clk : inst_to_clock_indexes.at(inst_id)) {
        if (clock_available_clock_region[clk][cr_id].template item<int32_t>() != 1) return false;
        if (!sb.isAvail.at(clk)) return false;
    }
    return true;
}

template<typename T>
void MinCostFlowLegalizer<T>::calculate_half_column_scoreboard_cost(at::Tensor pos) {

    for (auto &sb : hcsb_array) { std::fill(sb.clockCost.begin(), sb.clockCost.end(), 0); }
    auto pos_acc = pos.template accessor<T, 2>();   //

    // double baseCost = (double)_siteMap.size() / (double) _numSiteSLICE;
    // openparfPrint(MessageType::kDebug, "baseCost is: %f\n", baseCost);
    for (auto &m : sssir_model_infos) {
        auto inst_id_acc                = m.inst_ids.template accessor<int32_t, 1>();
        auto inst_weight_acc            = m.inst_weights.template accessor<T, 1>();
        auto inst_compatiable_sites_acc = m.inst_compatiable_sites.template accessor<T, 2>();
        for (int32_t i = 0; i < m.num_insts; i++) {
            int32_t inst_id = inst_id_acc[i];
            T       x       = pos_acc[inst_id][0];
            T       y       = pos_acc[inst_id][1];
            // DSPs and RAMs are similar, so we set the cost to 1 for now
            // TODO: cost setting for heterogeneous instances?
            double cost   = 1;
            auto   hc_idx = xy_to_half_column_functor(x, y);
            auto & sb     = hcsb_array.at(hc_idx);
            for (auto &clk : inst_to_clock_indexes[inst_id]) { sb.clockCost.at(clk) += cost; }
            if (hc_idx == 960) {
                openparfPrint(kDebug, "id %i hc %i site_x %f y %f\n", inst_id, hc_idx, x, y);
                for (auto &clk : inst_to_clock_indexes[inst_id]) {
                    std::cerr << clk << " " << sb.clockCost.at(clk);
                }
                std::cerr << std::endl;
            }
        }
    }
}

template<typename T>
std::vector<std::pair<int32_t, int32_t>>
MinCostFlowLegalizer<T>::check_half_column_sat_and_prune_clock() {
    constexpr uint32_t                       INDEX_TYPE_MAX = 1000000000;
    std::vector<std::pair<int32_t, int32_t>> pruned_hc_id;
    for (uint32_t hcId = 0; hcId < hcsb_array.size(); hcId++) {
        auto &sb = hcsb_array.at(hcId);
        // count the number of clk in this hc
        uint32_t cnt          = 0;
        uint32_t pruneClockId = INDEX_TYPE_MAX;
        uint32_t minCost      = INDEX_TYPE_MAX;
        for (uint32_t i = 0; i < sb.clockCost.size(); i++) {
            openparfAssertMsg(sb.isAvail[i] or (std::abs(sb.clockCost[i]) < 1e-6),
                              "Clk %i is in hc %i when it's not available.", i, hcId);
            if (sb.isFixed[i] or sb.clockCost[i] > 1e-6) { cnt++; }
            if (not sb.isFixed[i] and sb.clockCost[i] > 1e-6) {
                if (sb.clockCost[i] < minCost) {
                    minCost      = sb.clockCost[i];
                    pruneClockId = i;
                }
            }
        }
        if (cnt <= max_clock_net_per_half_column) continue;
        openparfAssertMsg(pruneClockId != INDEX_TYPE_MAX,
                          "All clock are fixed and none can be pruned. This should not happen.");
        // If clock number exceeds what is allowed, we disable the clock with the lowest non zero
        // cost
        openparfPrint(MessageType::kDebug, "prune clk %i in hc region %i\n", pruneClockId, hcId);
        pruned_hc_id.push_back(std::make_pair(pruneClockId, hcId));
        sb.isAvail.at(pruneClockId) = 0;
    }
    return std::move(pruned_hc_id);
}


template<typename T>
void MinCostFlowLegalizer<T>::calculate_max_distance(at::Tensor pos) {
    max_inst_to_site_distance.resize(0);
    dist_incr_per_iter.resize(0);
    auto pos_acc = pos.template accessor<T, 2>();   //
    for (auto &m : sssir_model_infos) {
        auto inst_id_acc                = m.inst_ids.template accessor<int32_t, 1>();
        auto inst_weight_acc            = m.inst_weights.template accessor<T, 1>();
        auto inst_compatiable_sites_acc = m.inst_compatiable_sites.template accessor<T, 2>();

        T min_px = std::numeric_limits<T>::max();
        T min_py = std::numeric_limits<T>::max();
        T max_px = std::numeric_limits<T>::lowest();
        T max_py = std::numeric_limits<T>::lowest();
        // Range of instances
        for (int32_t i = 0; i < m.num_insts; i++) {
            int32_t inst_id = inst_id_acc[i];
            T       px      = pos_acc[inst_id][0];
            T       py      = pos_acc[inst_id][1];
            min_px          = std::min(min_px, px);
            min_py          = std::min(min_py, py);
            max_px          = std::max(max_px, px);
            max_py          = std::max(max_py, py);
        }
        // Range of sites
        T min_sx = std::numeric_limits<T>::max();
        T min_sy = std::numeric_limits<T>::max();
        T max_sx = std::numeric_limits<T>::lowest();
        T max_sy = std::numeric_limits<T>::lowest();
        for (int32_t i = 0; i < m.num_sites; ++i) {
            T sx   = (inst_compatiable_sites_acc[i][0] + inst_compatiable_sites_acc[i][2]) / 2;
            T sy   = (inst_compatiable_sites_acc[i][1] + inst_compatiable_sites_acc[i][3]) / 2;
            min_sx = std::min(min_sx, sx);
            min_sy = std::min(min_sy, sy);
            max_sx = std::max(max_sx, sx);
            max_sy = std::max(max_sy, sy);
        }
        // Lower bound of maximum Manhatten distance between instances and sites of this type
        T max_dist = std::max(max_px, max_sx) - std::min(min_px, min_sx)    // x distance
                   + std::max(max_py, max_sy) - std::min(min_py, min_sy);   // y distance
        m.max_distance        = (double) max_dist;
        int32_t iteration_num = std::max((int32_t) (m.num_sites / site_per_iteration), 1);
        m.dist_incr_per_iter  = max_dist / iteration_num;
        ;
    }
}

template<typename T>
void MinCostFlowLegalizer<T>::init_clock_constraints() {
    for (uint32_t i = 0; i < num_half_column_regions; i++) {
        hcsb_array.emplace_back(num_clock_nets);
    }
    // Make all clock available at first
    for (auto &sb : hcsb_array) { std::fill(sb.isAvail.begin(), sb.isAvail.end(), 1); }
}


OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // TODO: how to support multiple pos.dtype?
    // Should we add another wrapper?
    py::class_<OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>>(m, "MinCostFlowLegalizer")
            .def(py::init<OPENPARF_NAMESPACE::database::PlaceDB const &>())
            .def("add_sssir_instances",
                 &OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>::add_sssir_instances,
                 "Add SSSIR instances")
            .def("forward", &OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>::forward,
                 "Min-cost flow legalization forward")
            .def("set_honor_clock_constraints",
                 &OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>::set_honor_clock_constraints,
                 "Set whether to consider clock region constraints")
            .def("set_slr_aware_flag",
                 &OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>::set_slr_aware_flag,
                 "Set whether to consider multi-die architecture")
            .def("set_max_clk_per_half_column",
                 &OPENPARF_NAMESPACE::MinCostFlowLegalizer<double>::set_max_clk_per_half_column,
                 "Set maximum number of clock nets in each half column")
            .def("reset_clock_available_clock_region",
                 &OPENPARF_NAMESPACE::MinCostFlowLegalizer<
                         double>::reset_clock_available_clock_region,
                 "Reset the clock available clock region");
}
