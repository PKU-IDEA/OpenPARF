#ifndef _MCF_H_
#define _MCF_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>

using namespace std;

#define MAX_DEGREE 40
namespace router {
// MCF: "max commodity flow"; MCMCF: "min-cost max concurrent flow"
enum PROBLEM_TYPE { MCF_TYPE = 0, MCMCF_TYPE = 1 };

////////////////////////////////////////////////////////////////////////////////
//
// NODE
//
////////////////////////////////////////////////////////////////////////////////

class NODE
{
 private:

 public:
    int id; // start from 0
    int x, y; // location coordinates directly in um;
    int pre; // parent node in shortest path tree
    int pre_edge; // parent edge in shortest path tree
    double dist; // distance for shortest path algorithm
    int no_comm; // number of destinations for commodities starting from this node
    int *comms; // list of commodities starting from the node
    int no_edge; // number of edges incident to the node
    int *edges; // list of edges incident to the node
    int dij_visited; // flag for Dijkstra algo
    int dij_updated; // second flag for Dijkstra algo
    int min_visited; // flag for searching min c(e)
    // _preferred_path has a number of elements equal to the number of 
    // commodities; each entry stores the next, downstream, node index
    // of the preferred unique path of the flow of this commodity from 
    // src toward des; this is constructed during the randomized rounding;
    int *_preferred_path;
 public:
    NODE() {}
    ~NODE() {}
};

////////////////////////////////////////////////////////////////////////////////
//
// EDGE
//
////////////////////////////////////////////////////////////////////////////////

class EDGE
{
 private:

 public:
    int id; // start from 0
    int src, dest; // source and destination node id
    double latency; // delay of this edge; will play role of cost;
    double length;
    // _flows has a number of elements equal to the number of demands/
    // commodities with the index being the id of demand;
    double *_flows;
    double flow; // accumulated total flow;
    double capacity; // c_e 
    double left_capacity;
    // dual of capacity;
    double _Y_e;
    double _old_Y_e;

 public:
    EDGE() {}
    ~EDGE() {}

    void set_flow_of_commodity( int id, double flow) { _flows[ id] = flow; }
    void add_to_flow_of_commodity( int id, double val) { _flows[ id] += val; }
    double flow_of_commodity( int id) { return _flows[ id]; }
};

////////////////////////////////////////////////////////////////////////////////
//
// COMMODITY
//
////////////////////////////////////////////////////////////////////////////////

class COMMODITY
{
 public:
    int id; // start from 0
    int src, dest; // source and destination 
    double demand;
    double left_demand;

 public:
    COMMODITY() {}
    ~COMMODITY() {}
};

////////////////////////////////////////////////////////////////////////////////
//
// MCF acts as a host to the graph and methods such as Dijkstra and solver;
//
////////////////////////////////////////////////////////////////////////////////

class MCF
{
 private:
    // Note: "inherited" like this from the original code; I should
    // make them vectors;
    int no_node;
    NODE *nodes;
    int no_edge;
    EDGE *edges;
    int no_cut;
    int no_commodity;
    COMMODITY *_commodities;

    // primal variables are the actual final flows through edges of graph;
    // their values are stored in _flows of EDGE class;
    // dual variable PHI_d; the other dual variables are Y_e
    // and the "length function" (see Ababei paper);
    double _phi_latency;
    // L is used to record the minimum latency achievable; utilized as
    // budget in the MCF problem formulation; found out via binary search;
    double L, LL, UL; // L is latency budget; LL/UP is lower/upper latency;

    // lambda_max is the first lambda that mcf() returns, with initial
    // latency_limit relaxed to 1000000 (inf);
    double _lambda_max;
    double _latency_max; // associated with lambda_max;
    double _total_latency;

    // control variables;
    double _delta;
    double _epsilon1;
    double _scale;
    // s = [P(lambda_max) - L]/[lambda_max - dual]; see eq. 9 of Hu
    // paper; used to implement "interval estimation";
    double _s;
    // temp_edge_flow stores how much flow is routed during an iteration;
    // it is a sketch array;
    double *_temp_edge_flow;
    int _rd;
    int _min_rd;
 public:
    // arguments;
    PROBLEM_TYPE _problem_type;
    char _network_filename[512];

 public:
    MCF() {
        no_node = 0;
        no_edge = 0;
        no_commodity = 0;

        L=0; LL=0; UL=0;
        _phi_latency = 0;
        _lambda_max = 0;
        _latency_max = 0;
        _total_latency = 0;

        _delta = 1.0;
        _epsilon1 = 0.9;
        _scale = 1;
        _s = -1;
        _min_rd = 0;
        _rd = 0;
        _temp_edge_flow = 0;
        _problem_type = MCMCF_TYPE;
    }
    ~MCF() {}

    // utils;
    double flow_of_commodity_thru_edge( int c_id, int e_id) {
        //assert(c_id >= 0 && c_id < no_commodity);
        //assert(e_id >= 0 && e_id < no_edge);
        return edges[ e_id].flow_of_commodity( c_id);
    }
    double get_L() const { return L; }
    int problem_type() const { return _problem_type; }

    // host;
    void initialize( double delta, int flag);
    bool parse_options( int argc, char **argv);
    bool feasibility_check();
    double compute_D();
    double compute_alpha();
    double compute_lambda();
    void route_flow( int t, double amount, int commodity_id);
    double check_latency_constraint( int t);
    double min_capacity( int s);
    double min_capacity_this_commodity( int dest);
    void reset_left_capacities_in_tree( int s);
    void update_dual_variables( int s, double epsilon, int flag);
    double minimum( double x, double y, double z);
    void scale_down_linear( float times);
    void scale_down_flows( int phase_count);

    // Dijkstra;
    void shortest_paths(int s, int num_commodities, int *dest_flag);

    // MCF solver;
    void init_param();
    int run_mcf_solver();
    double mcf( double delta, double epsilon, int flag);
    bool do_randomized_rounding();

    // graph related utilities;
    bool build_network_from_file(double latency_limit=1000000, double rate=1);
    bool build_network_from_host_application();
    void read_network_topology_and_demands( FILE *fp, double rate=1);
    void free_topology();
    void print_network_demands( bool print_only_edges = false);
    void print_backward_shortest_path( int t);
    void print_routing_paths(); 
   
   friend class LocalRouter;
};
}
#endif
