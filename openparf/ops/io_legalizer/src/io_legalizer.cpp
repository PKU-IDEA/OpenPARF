/**
 * File              : io_legalizer.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 08.26.2021
 * Last Modified Date: 08.26.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "io_legalizer.h"

#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>

OPENPARF_BEGIN_NAMESPACE

using Site       = database::Site;
using Resource   = database::Resource;

using SiteRefVec = std::vector<std::reference_wrapper<const Site>>;

int32_t SelectResource(database::PlaceDB const &placedb, int32_t area_type_id) {
  int32_t num_resources   = placedb.numResources();
  int32_t selected_rsc_id = -1;
  for (int32_t rsc_id = 0; rsc_id < num_resources; rsc_id++) {
    const auto &area_type_ids = placedb.resourceAreaTypes(Resource(rsc_id));
    bool is_contained = std::find(area_type_ids.begin(), area_type_ids.end(), area_type_id) != area_type_ids.end();
    if (is_contained) {
      openparfAssert(selected_rsc_id == -1);
      selected_rsc_id = rsc_id;
    }
  }
  return selected_rsc_id;
}

SiteRefVec CollectBoxes(database::PlaceDB const &placedb, int32_t rsc_id) {
  auto const &layout = placedb.db()->layout();
  SiteRefVec  concerned_sites;
  for (const Site &site : layout.siteMap()) {
    auto const &site_type = layout.siteType(site);
    if (site_type.resourceCapacity(rsc_id) > 0) {
      concerned_sites.emplace_back(site);
    }
  }
  return std::move(concerned_sites);
}

template<class T>
void DispatchedIoLegalizerForward(database::PlaceDB const &placedb,
                                  int32_t                  num_concerned_insts,
                                  T                       *pos,
                                  T                       *pos_xyz,
                                  int32_t                 *concerned_inst_ids,
                                  int32_t                  area_type_id) {
  auto const &layout              = placedb.db()->layout();
  auto const &place_params        = placedb.place_params();
  int32_t     rsc_id              = SelectResource(placedb, area_type_id);
  SiteRefVec  concerned_sites     = CollectBoxes(placedb, rsc_id);
  int32_t     num_concerned_sites = concerned_sites.size();

  openparfPrint(kDebug, "area type: %s(%d)\n", place_params.area_type_names_[area_type_id].c_str(), area_type_id);
  openparfPrint(kDebug, "resouce type: %d\n", rsc_id);
  openparfPrint(kDebug, "#inst with area type %d: %d\n", area_type_id, num_concerned_insts);
  openparfPrint(kDebug, "#sites with resource %d: %d\n", rsc_id, num_concerned_sites);


  // min-cost flow-based legalization
  using GraphType    = lemon::ListDigraph;
  using CapacityType = int32_t;
  using CostType     = int64_t;
  using SolverType   = lemon::NetworkSimplex<GraphType, CapacityType, CostType>;
  using ResultType   = typename SolverType::ProblemType;

  GraphType                                                 graph;
  GraphType::ArcMap<CapacityType>                           capacity_lower_bound(graph), capacity_upper_bound(graph);
  GraphType::ArcMap<CostType>                               cost(graph);
  SolverType                                                solver(graph);
  GraphType::Node                                           source, drain;
  std::vector<GraphType::Node>                              instance_nodes, site_nodes;
  std::vector<GraphType::Arc>                               instance_arcs, site_arcs;
  std::vector<std::tuple<int32_t, int32_t, GraphType::Arc>> inst_to_site_arcs;
  std::vector<int32_t>                                      concerned_site_io_count;

  source = graph.addNode();
  drain  = graph.addNode();

  instance_nodes.clear();
  instance_arcs.clear();
  CapacityType total_supply = 0;
  for (int32_t i = 0; i < num_concerned_insts; i++) {
    instance_nodes.emplace_back(graph.addNode());
    auto &node = instance_nodes.back();
    instance_arcs.emplace_back(graph.addArc(source, node));
    auto &arc                 = instance_arcs.back();
    capacity_lower_bound[arc] = 0;
    capacity_upper_bound[arc] = 1;
    cost[arc]                 = 0;
    total_supply += 1;
  }

  site_nodes.clear();
  site_arcs.clear();
  CapacityType total_capacity = 0;
  for (int32_t i = 0; i < num_concerned_sites; i++) {
    const Site &site         = concerned_sites[i];
    int32_t     rsc_capacity = placedb.getSiteCapacity(site, rsc_id);
    if (i == 0) {
      openparfPrint(kDebug, "site capacity of resource %d: %d\n", rsc_id, rsc_capacity);
    }
    site_nodes.emplace_back(graph.addNode());
    auto &node = site_nodes.back();
    site_arcs.emplace_back(graph.addArc(node, drain));
    auto &arc                 = site_arcs.back();
    capacity_lower_bound[arc] = 0;
    capacity_upper_bound[arc] = rsc_capacity;
    cost[arc]                 = 0;
    total_capacity += rsc_capacity;
  }

  openparfPrint(kDebug, "total Supply: %d\n", total_supply);
  openparfPrint(kDebug, "total Capacity: %d\n", total_capacity);
  openparfAssert(total_supply <= total_capacity);

  inst_to_site_arcs.clear();
  for (int i = 0; i < num_concerned_insts; i++) {
    for (int j = 0; j < num_concerned_sites; j++) {
      int32_t     inst_id       = concerned_inst_ids[i];
      auto        instance_node = instance_nodes[i];
      const Site &site          = concerned_sites[j];
      auto        site_node     = site_nodes[j];

      T           xx            = pos[inst_id << 1];
      T           yy            = pos[inst_id << 1 | 1];
      T           site_x        = (site.bbox().xl() + site.bbox().xh()) * 0.5;
      T           site_y        = (site.bbox().yl() + site.bbox().yh()) * 0.5;
      CostType    dist          = (std::fabs(site_x - xx) + std::fabs(site_y - yy)) * 100;

      inst_to_site_arcs.emplace_back(i, j, graph.addArc(instance_node, site_node));
      auto &arc                 = std::get<2>(inst_to_site_arcs.back());
      capacity_lower_bound[arc] = 0;
      capacity_upper_bound[arc] = 1;
      cost[arc]                 = dist;
    }
  }

  solver.reset();
  auto rv = solver.stSupply(source, drain, total_supply)
                    .lowerMap(capacity_lower_bound)
                    .upperMap(capacity_upper_bound)
                    .costMap(cost)
                    .run();

  CapacityType total_flow = 0;
  for (auto const &arc : site_arcs) total_flow += solver.flow(arc);
  openparfAssert(rv == ResultType::OPTIMAL);
  openparfAssert(total_flow == total_supply);

  concerned_site_io_count.resize(num_concerned_sites, 0);
  for (const auto &inst_site_arc : inst_to_site_arcs) {
    int32_t     inst_id = concerned_inst_ids[std::get<0>(inst_site_arc)];
    const Site &site    = concerned_sites[std::get<1>(inst_site_arc)];
    auto        arc     = std::get<2>(inst_site_arc);
    if (solver.flow(arc) > 0) {
      T site_x                 = (site.bbox().xl() + site.bbox().xh()) * 0.5;
      T site_y                 = (site.bbox().yl() + site.bbox().yh()) * 0.5;
      pos[inst_id << 1]        = site_x;
      pos[inst_id << 1 | 1]    = site_y;
      pos_xyz[inst_id * 3]     = site_x;
      pos_xyz[inst_id * 3 + 1] = site_y;
      pos_xyz[inst_id * 3 + 2] = concerned_site_io_count[std::get<1>(inst_site_arc)]++;
    }
  }
}

void IoLegalizerForward(database::PlaceDB const &placedb,
                        at::Tensor               pos,
                        at::Tensor               pos_xyz,
                        at::Tensor               concerned_inst_ids,
                        int32_t                  area_type_id) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(pos_xyz);
  CHECK_CONTIGUOUS(pos_xyz);
  CHECK_FLAT_CPU(concerned_inst_ids);
  CHECK_CONTIGUOUS(concerned_inst_ids);
  AT_ASSERTM(concerned_inst_ids.dtype() == torch::kInt32, "`concerned_inst_ids` must be a Int32 tensor.");
  int num_concerned_insts = concerned_inst_ids.numel();
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "IoLegalizerForward", [&] {
    DispatchedIoLegalizerForward(placedb,
                                 num_concerned_insts,
                                 OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                 OPENPARF_TENSOR_DATA_PTR(pos_xyz, scalar_t),
                                 OPENPARF_TENSOR_DATA_PTR(concerned_inst_ids, int32_t),
                                 area_type_id);
  });
}

template void DispatchedIoLegalizerForward<float>(database::PlaceDB const &placedb,
                                                  int32_t                  num_concerned_insts,
                                                  float                   *pos,
                                                  float                   *pos_xyz,
                                                  int32_t                 *concerned_inst_ids,
                                                  int32_t                  area_type_id);
template void DispatchedIoLegalizerForward<double>(database::PlaceDB const &placedb,
                                                   int32_t                  num_concerned_insts,
                                                   double                  *pos,
                                                   double                  *pos_xyz,
                                                   int32_t                 *concerned_inst_ids,
                                                   int32_t                  area_type_id);

OPENPARF_END_NAMESPACE
