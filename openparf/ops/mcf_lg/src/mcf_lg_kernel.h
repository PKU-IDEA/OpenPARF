
#include <functional>
#include <vector>

#include "database/placedb.h"
#include "lemon/list_graph.h"
#include "lemon/network_simplex.h"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE
template<typename T>
class MinCostFlowLegalizer {
 public:
  using GraphType    = lemon::ListDigraph;
  using CapacityType = int32_t;
  using CostType     = int32_t;
  using SolverType   = lemon::NetworkSimplex<GraphType, CapacityType, CostType>;

  struct MinCostFlowProblem {
    using ResultType = typename SolverType::ProblemType;
    GraphType                       graph;
    GraphType::ArcMap<CapacityType> capacity_lower_bound, capacity_upper_bound;
    GraphType::ArcMap<CostType>     cost;
    GraphType::Node                 source, drain;
    CapacityType                    supply;
    SolverType                      solver;
    MinCostFlowProblem() : capacity_lower_bound(graph), capacity_upper_bound(graph), cost(graph), solver(graph) {
      source = graph.addNode();
      drain  = graph.addNode();
    }

    void dump(std::string dump_file) {
      std::fstream s(dump_file, s.out);
      s << "ArcID, SourceNodeId, TargetNodeId, LowerCapacity, UpperCapacity, Cost" << std::endl;
      for (GraphType::ArcIt i(graph); i != lemon::INVALID; ++i) {
        s << graph.id(i) << ", " << graph.id(graph.source(i)) << ", " << graph.id(graph.target(i)) << ", ";
        s << capacity_lower_bound[i] << ", ";
        s << capacity_upper_bound[i] << ", ";
        s << cost[i] << std::endl;
      }
      s.close();
    }

    bool solve() {
      solver.reset();
      auto res = solver.stSupply(source, drain, supply)
                         .lowerMap(capacity_lower_bound)
                         .upperMap(capacity_upper_bound)
                         .costMap(cost)
                         .run();

      if (res == ResultType::INFEASIBLE) {
        openparfPrint(MessageType::kError,
                      "Min cost flow problem has no feasible "
                      "solution. Not meaningful to continue\n");
        return false;
      } else if (res == ResultType::UNBOUNDED) {
        openparfPrint(MessageType::kError, "Min cost flow formulation is unbounded. Not meaningful to continue\n");
        return false;
      }
      openparfAssert(res == ResultType::OPTIMAL);
      return true;
    }
  };

  struct SSSIRModelInfo {
    int32_t                                                   num_insts;
    int32_t                                                   num_sites;
    std::vector<GraphType::Node>                              instance_nodes, site_nodes;
    std::vector<GraphType::Arc>                               instance_arcs, site_arcs;
    std::vector<std::tuple<int32_t, int32_t, GraphType::Arc>> inst_to_site_arcs;
    at::Tensor                                                inst_ids;
    at::Tensor                                                inst_weights;
    at::Tensor                                                inst_compatiable_sites;
    double                                                    max_distance;
    double                                                    dist_incr_per_iter;
    int32_t                                                   model_id;
    std::vector<std::vector<int32_t>>                         hc_arcs;
  };
  struct HalfColumnRegionScoreBoard {
    explicit HalfColumnRegionScoreBoard(uint32_t numClockNets)
        : isAvail(numClockNets, 1),
          clockCost(numClockNets, 0.0),
          isFixed(numClockNets, 0) {}
    // All the clock are clock net indexes, not clock net id.
    std::vector<uint8_t> isAvail;     // Record if a clock (index) is allowed in this HC
    std::vector<double>  clockCost;   // The cost of forbidding a clock in this HC
    std::vector<uint8_t> isFixed;     // isFixed[i] records if a fix inst (i.e. a previously
                                      // legalized DSP/RAM) is in this HC and it contains clk i
  };

  MinCostFlowLegalizer(database::PlaceDB const &placedb)
      : _placedb(placedb),
        inst_to_clock_indexes(placedb.instToClocks()) {
    site_per_iteration         = 100;
    scale_factor               = 1000;
    num_clock_nets             = placedb.numClockNets();
    num_half_column_regions    = placedb.numHalfColumnRegions();
    xy_to_clock_region_functor = [&](int32_t x, int32_t y) {
      return placedb.XyToCrIndex((database::PlaceDB::CoordinateType) x, (database::PlaceDB::CoordinateType) y);
    };
    xy_to_half_column_functor = [&](int32_t x, int32_t y) {
      return placedb.XyToHcIndex((database::PlaceDB::CoordinateType) x, (database::PlaceDB::CoordinateType) y);
    };
  }
  at::Tensor forward(at::Tensor pos);
  void       add_sssir_instances(at::Tensor inst_ids, at::Tensor inst_weights, at::Tensor inst_compatiable_sites);
  void       set_honor_clock_constraints(bool v) { honor_clock_region_constraints = v; }
  void       reset_clock_available_clock_region(at::Tensor _clock_available_clock_region) {
    clock_available_clock_region = _clock_available_clock_region;
  }
  void set_max_clk_per_half_column(int c) {
    openparfAssert(c > 0);
    max_clock_net_per_half_column = c;
  }

 private:
  void                                      calculate_max_distance(at::Tensor pos);

  void                                      init_clock_constraints();
  bool                                      inst_clock_legal_at_site(int32_t inst_id, int32_t site_x, int32_t site_y);
  void                                      calculate_half_column_scoreboard_cost(at::Tensor pos);
  std::vector<std::pair<int32_t, int32_t>>  check_half_column_sat_and_prune_clock();

  std::vector<SSSIRModelInfo>               sssir_model_infos;
  MinCostFlowProblem                        mcf_problem;

  std::vector<T>                            max_inst_to_site_distance;
  std::vector<T>                            dist_incr_per_iter;
  int32_t                                   site_per_iteration;
  int32_t                                   scale_factor;

  // Clock constraint related structure
  bool                                      honor_clock_region_constraints;
  std::function<int32_t(int32_t, int32_t)>  xy_to_half_column_functor;
  std::function<int32_t(int32_t, int32_t)>  xy_to_clock_region_functor;
  const std::vector<std::vector<uint32_t>> &inst_to_clock_indexes;
  at::Tensor                                clock_available_clock_region;
  int32_t                                   num_clock_nets;
  int32_t                                   num_half_column_regions;
  // Half column structures
  std::vector<HalfColumnRegionScoreBoard>   hcsb_array;
  int32_t                                   max_clock_net_per_half_column;

  // I give up, binding is too complicated
  database::PlaceDB const                  &_placedb;
};

OPENPARF_END_NAMESPACE