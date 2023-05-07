#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "pqueue.h"
#include "mcf.h"
#include <iostream>

using namespace std;
namespace router {
////////////////////////////////////////////////////////////////////////////////
//
// MCF host
//
////////////////////////////////////////////////////////////////////////////////

void MCF::initialize( double delta, int flag)
{
    // called each time mcf() is called;
    int i=0;

    // init dual variables
    if ( flag == 0) { // 0: "max concurrent flow"
        _phi_latency = 0.0;
    } else { // 1 "min-cost max concurrent flow"
        _phi_latency = delta / L; // dual variable PHI_d = 1/1000000
    }
    for ( i = 0; i < no_edge; i++) {
        edges[i]._Y_e = delta / edges[i].capacity;
        edges[i]._old_Y_e = edges[i]._Y_e;
    }
    // init edges
    for ( i = 0; i < no_edge; i++) {
        edges[i].flow = 0.0;
        for ( int j = 0; j < no_commodity; j++) {
            edges[i]._flows[ j] = 0.0;
        }
    }
    // reset edge flows
    for ( i = 0; i < no_edge; i++) {
        _temp_edge_flow[i] = 0.0;
    }   
    // init edge "length function" l(e)
    for ( i = 0; i < no_edge; i++) {
        edges[i].length = 0.0;
        edges[i].length += edges[i]._Y_e;
        edges[i].length += edges[i].latency * _phi_latency; // 0 for flag=0;
    }
    // init commodities
    for ( i = 0; i < no_commodity; i++) {
        _commodities[i].left_demand = _commodities[i].demand;
    }

    // reset _total_latency, which will be computed as the summation
    // of individual latencies from shortest-path trees for each source
    // of commodities;
    _total_latency = 0.0;
}

bool MCF::parse_options( int argc, char **argv)
{
    // Note: if this code is to be used within a host code (i.e., not
    // as a stand alone tool) then, the stuff in here should be done
    // from with the host code;

    // (1) parse command line arguments;
    // return false if error
    if ( argc == 1) {
        printf("\nUsage:  mcf_solver network_file [Options...]\n");
        printf("Options:\n");
        printf("\t[-problem_type MCF|MCMCF]. Default is MCMCF.\n");
        printf("\twhere: MCF - max multicommodity flow, MCMCF - min-cost max concurrent flow\n");
        printf("\t[-epsilon float]. Default is 0.1.\n");
        exit(1);
    }
    // first argument is always the network file;
    sprintf( _network_filename, "%s", argv[1]);
    _problem_type = MCMCF_TYPE; // default;
    _epsilon1 = 0.1; // default;

    int i = 2;
    while ( i < argc) {

        if ( strcmp (argv[i],"-problem_type") == 0) {
            if (argc <= i+1) {
                printf("Error:  -problem_type option requires a string parameter.\n");
                exit(1);
            } 
            if (strcmp(argv[i+1], "MCF") == 0) {
                _problem_type = MCF_TYPE;
            } 
            else if (strcmp(argv[i+1], "MCMCF") == 0) {
                _problem_type = MCMCF_TYPE;
            } else {
                printf("Error:  -problem_type must be MCF or MCMCF.\n");
                exit(1);
            }
            i += 2;
            continue;
        }

        if ( strcmp(argv[i], "-epsilon") == 0) {
            _epsilon1 = atof(argv[i+1]);
            if ( _epsilon1 <= 0 || _epsilon1 >= 1) {
                printf("Error:  -epsilon option requires a float in (0,1).\n");
                exit(1);
            }
            i += 2;
            continue;
        }
    }

    return true;
}

void MCF::init_param()
{
    // called one time only from inside build_network_from_file() because
    // we need the number of edges of the graph for delta calculation;
    // () set latency budget to infinity (inf);
    L = 1000000.0;
    // () epsilon is now set to default _epsilon1 = 0.1 inside parse_options();
    // or it could be set by user via command line argument;
    // () delta is set according to equation 3 from Karakostas paper;
    double epsilon = _epsilon1;
    _delta = (1/pow(1+epsilon, (1-epsilon)/epsilon))*(pow((1-epsilon)/no_edge, 1/epsilon));
    // () expected number of iterations (or phases) of the outer loop;
    // currently it is not used for any purpose;
    _scale = log((1+epsilon)/_delta) / log(1+epsilon);
    //printf("\nepsilon=%e delta=%e _scale=%e\n",_epsilon1,_delta,_scale); // exit(1);
}

bool MCF::feasibility_check()
{
    // check and see if the routed flows violate capacities; if so,
    // then return false: no feasible solution; this is a "stretch";
    // feasibility should be checked differently;
    double threshold, violation;
    bool printed_warning = false;
    for ( int i = 0; i < no_edge; i++) {
        if ( edges[i].flow > edges[i].capacity) {
            // consider only violations that are greater than 3 * epsilon;
            threshold = 3 * _epsilon1 * edges[i].capacity;
            violation = (edges[i].flow - edges[i].capacity);
            if ( violation > threshold) {
                return false;
            } else {
                // print once only a warning;
                if ( !printed_warning) {
                    printf("\nWarning:  Some edges have capacity violation within 3*epsilon");
                    printed_warning = true;
                }
            }
        }
    }
    return true; // solution is ok;
}

double MCF::compute_D()
{
    // "D" is the numerator of dual=D/alpha; see section 6 of Garg paper;
    double D = 0.0; 
    for ( int i = 0; i < no_edge; i++) {
        D += edges[i]._Y_e * edges[i].capacity;
    }
    D += L * _phi_latency;

    return D;
}

double MCF::compute_alpha()
{
    // "alpha" is the denuminator of dual=D/alpha; see section 6 of Garg paper;
    int i, j;
    double alpha = 0.0; // to return;
    for ( i = 0; i < no_node; i++) {
        if ( nodes[i].no_comm) {
            int *dest_flag = (int*)malloc((no_node)*sizeof(int));
            if ( dest_flag == NULL) {
                printf("\nError: Unable to malloc <getAlpha>.\n"); exit (1);
            }
            memset((void*)dest_flag,0,(no_node)*sizeof(int));
                
            for ( j = 0; j < nodes[i].no_comm; j++) {
                dest_flag[_commodities[nodes[i].comms[j]].dest] = 1;
            }
            
            shortest_paths( nodes[i].id, nodes[i].no_comm, dest_flag);
            _rd++;
            free( dest_flag);

            for ( j = 0; j < nodes[i].no_comm; j++) {
                alpha += _commodities[nodes[i].comms[j]].demand * 
                    nodes[_commodities[nodes[i].comms[j]].dest].dist;
            }
        }
    }

    return alpha;
}

double MCF::compute_lambda()
{
    // compute lambda=MIN(actual flow/demand) among all commodities;
    double lambda = DBL_MAX;

    for ( int comm_i = 0; comm_i < no_commodity; comm_i++) {
        // for each commodity we take its source node and look
        // at its outgoing edges to sum all flow pushed/routed 
        // for this commodity;
        int src_id = _commodities[comm_i].src; // source node;
        double routed_flow_this_commodity = 0.0;
        for ( int j = 0; j < nodes[src_id].no_edge; j++) {
            int edge_id = nodes[src_id].edges[j];
            routed_flow_this_commodity += edges[edge_id]._flows[ comm_i];
        }
        double this_lambda = routed_flow_this_commodity / _commodities[comm_i].demand;
        if ( this_lambda < lambda) {
            lambda = this_lambda;
        }
    }

    return lambda;
}

double MCF::check_latency_constraint( int dest)
{
    // this is L/c(P) in Fleischer paper (pp. 10), where
    // c(P) is is the cost of sending one unit of flow along
    // the shortest path: Sum_{e in P}{D(e)}, where D(e) is
    // latency of each edge along path;
    int t = dest;
    double cost_to_send_unit_flow = 0.0; // along the shortest path to this dest;
    while ( nodes[t].pre != -1) {
        cost_to_send_unit_flow += edges[nodes[t].pre_edge].latency;
        t = nodes[t].pre;
    }
    
    return L/cost_to_send_unit_flow;
}

double MCF::min_capacity( int s) 
{
    // Note: currently not used;
    // find "c" as the minimum capacity of the edges on ALL
    // the paths in the shortest paths tree for this source node "s";
    int t = 0;
    double min_capacity = 1000000.0;

    _min_rd++;
    // start from all dest nodes, traverse shortest path tree;
    for ( int i = 0; i < nodes[s].no_comm; i++) {
        if ( _commodities[nodes[s].comms[i]].left_demand > 1e-3) {
            // pick up this destination and walk backward to sourse "s";
            t = _commodities[nodes[s].comms[i]].dest;
            while ( (nodes[t].pre != -1) && (nodes[t].min_visited != _min_rd)) {
                int edge_id = nodes[t].pre_edge;
                nodes[t].min_visited = _min_rd;

                if ( edges[edge_id].capacity < min_capacity) {
                    min_capacity = edges[edge_id].capacity;
                }
            }
        }
    }
    return min_capacity;
}

double MCF::min_capacity_this_commodity( int dest) 
{
    // find "c" as the minimum available capacity of the edges on 
    // the shortest path for this sink node "t";
    double min_avail_capacity = 1000000.0;

    int t = dest;
    while ( nodes[t].pre != -1) {
        int edge_id = nodes[t].pre_edge;
        if ( edges[edge_id].left_capacity < min_avail_capacity) {
            min_avail_capacity = edges[edge_id].left_capacity;
        }
        t = nodes[t].pre;
    }
    return min_avail_capacity;
}

void MCF::reset_left_capacities_in_tree( int s)
{
    // reset left_capacities of edges in the shortest paths tree to the
    // initial capacities; u'(e)=u(e), for any e in tree;
    int t = 0;
    // start from all dest nodes, traverse shortest path tree;
    for ( int i = 0; i < nodes[s].no_comm; i++) {
        if ( _commodities[nodes[s].comms[i]].left_demand > 1e-3) {
            // pick up this destination and walk backward to sourse "s";
            t = _commodities[nodes[s].comms[i]].dest;
            while ( nodes[t].pre != -1) {
                int edge_id = nodes[t].pre_edge;
                edges[edge_id].left_capacity = edges[edge_id].capacity;
                t = nodes[t].pre;
            }
        }
    }
}

void MCF::route_flow( int t, double routed_amount, int commodity_id)
{
    // t is destination to which we route "amount" of commodity;
    while ( nodes[t].pre != -1) {
        int edge_id = nodes[t].pre_edge;
        _temp_edge_flow[edge_id] += routed_amount;
        edges[edge_id].left_capacity -= routed_amount;
        
        // record this routed_amount for this commodity id on the 
        // corresponding edge also;
        assert(commodity_id >= 0 && commodity_id < no_commodity);
        edges[ edge_id]._flows[ commodity_id] += routed_amount;

        t = nodes[t].pre;
    }
    return;
}

void MCF::update_dual_variables( int s, double epsilon, int flag)
{
    // update dual variables; compute l_i_j_s(e), where
    // "j" is jth iteration of phase "i", and "s" is the current step;
    int i, t;
    double old_phi_latency;
    double temp_latency = 0.0;

    // (1) accumulate temp_latency along the shortest paths for the
    // shortest paths tree for the commodities of this source node;
    _min_rd++;
    for ( i = 0; i < nodes[s].no_comm; i++) {
        t = _commodities[nodes[s].comms[i]].dest;
        while ( (nodes[t].pre != -1) && (nodes[t].min_visited != _min_rd)) {
            int edge_id = nodes[t].pre_edge;
            nodes[t].min_visited = _min_rd;
            
            temp_latency += _temp_edge_flow[edge_id] * edges[edge_id].latency;

            // update the dual variable Y_e;
            edges[edge_id]._old_Y_e = edges[edge_id]._Y_e;
            // Note: _temp_edge_flow[edge_id] represents the amount of total
            // flow of all commodities that have the same source "s", which is
            // pushed thru this edge during this step "s";
            edges[edge_id]._Y_e *= 
                (1 + epsilon * _temp_edge_flow[edge_id] / edges[edge_id].capacity);

            // walk upstream on shortest path;
            t = nodes[t].pre;
        }
    }
    _min_rd++;
    // record latency contributed due to total flow pushed thru during
    // this step "s";
    _total_latency += temp_latency;

    // (2) update additional dual variable PHI_d;
    old_phi_latency = _phi_latency;
    _phi_latency *= (1 + epsilon * temp_latency / L); // adjust value from prev. iter;

    // (3) update the "length function";
    for ( i = 0; i < no_edge; i++) {
        edges[i].length += (edges[i]._Y_e - edges[i]._old_Y_e);
        // the above length function is enough for "max concurrent flow" problem;
        // howver, if we solve "min-cost max concurrent flow", then, we must add
        // more to the length function;
        if ( flag != 0) { // 1
            edges[i].length += edges[i].latency * (_phi_latency - old_phi_latency);
        }
    }

    // (4) add to the flow recorded for each edge the accumulated 
    // amounts (as sum of f_{i,j,s}^{c_q}) for each commodity routed during
    // this iteration, amounts which are reflected by _temp_edge_flow (which
    // has values != zero) for edges of shortest path of this iter;
    for ( i = 0; i < no_edge; i++) {
        edges[i].flow += _temp_edge_flow[i];
    }

    // (5) reset temp storage of pushed flow during this iter; prepare it
    // for the next push/iteration;
    for ( i = 0; i < no_edge; i++) {
        _temp_edge_flow[i] = 0.0;
    }

    return;
}

void MCF::scale_down_linear( float times) 
{
    // Note: currently not used;
    for ( int i = 0; i < no_edge; i++) {
        edges[i].length /= times;
        edges[i]._Y_e /= times;
    }
    _phi_latency /= times;
    return;
}

void MCF::scale_down_flows( int phase_count)
{
    // scale down final solution; basically averaging over the number 
    // of phases (iterations of the main big loop of mcf);
    int scale = max( 1, phase_count); // this is "t";
    for ( int i = 0; i < no_edge; i ++) {
        edges[i].flow /= scale;
        for ( int j = 0; j < no_commodity; j ++) {
            edges[i]._flows[ j] /= scale;
        }
    }
}

double MCF::minimum( double x, double y, double z) 
{
    double min;
    if ( x < y) {
        if ( x < z) min = x;
        else min = z;
    } else {
        if ( y < z) min = y;
        else min = z;
    }
    return min;
}

////////////////////////////////////////////////////////////////////////////////
//
// MCF actual solver
//
////////////////////////////////////////////////////////////////////////////////

int MCF::run_mcf_solver()
{
    // it is assumed that the network was already created from file
    // or host application;


    // (1) first, run of MCF solver with the latency constraint
    // relaxed to infinity L=1000000 (inf); this is basically
    // the "max commodity flow" problem;
    // Reminder on MCF flavors:
    // -- "max multicommodity flow": total flow summed over all commodities
    //    is to be maximized;
    // -- "max concurrent flow": each commodity si,ti has a demand di; 
    //    objective is to maximize the fraction of the demand that can be shipped
    //    simultaneously for all commodities;
    // -- "min-cost max concurrent flow";
    printf("\nPART 1 - MAX CONCURRENT FLOW (MCF):");
    // flag=0 means that this is a "max commodity flow" run; there is
    // no latency constraint/budget;
    _lambda_max = mcf( _delta, _epsilon1, 0); // flag=0;
    //print_network_demands(true); // exit(1); // debug;

    // early exit if there is no "feasible" solution;
    if ( feasibility_check() == false) {
        printf("\nWarning: No feasible solution; some edges have capacity ");
        printf("\n         violation greater than 3*epsilon.\n");
        free_topology();
        exit(1);
    }

    // Note: at this time we could simply stop is we were not interested
    // in solving this problem such that the minimum latency is also achieved;
    // the minimum latency (stored in L) is found via binary search by
    // solving repeatedly the so called "min-cost max concurrent flow" problem;
    // also note that the solution we have now is most likely different
    // from the solution we'll have after the binary search;
    // so, if user wants a solution for the problem "max commodity flow" only,
    // then stop here;
    if ( _problem_type == MCF_TYPE) {
        return 1;
    }


    // (2) second, "improved" binary search to refine L; basically we look 
    // for the minimum latency achievable; during this search mcf is run with 
    // flag=1, that is as a "min-cost max concurrent flow";
    printf("\n\nPART 2 - BINARY SEARCH FOR L - MIN-COST MAX CONCURRENT FLOW (MCMCF):");
    // maximum latency is as resulted after finding the solution of the
    // "max multicommodity flow" problem from PART 1;
    _latency_max = _total_latency; // Hu: 1000000;
    LL = 0;
    UL = _total_latency; // Hu: _latency_max/_lambda_max;
    _s = -1;

    int counter = 0;
    while ( (UL - LL)/LL > 0.1) {
        // (a) set Latency as the middle point between LL and UL;
        L = (LL + UL) / 2;
        // (b) this call of MCF modifies LL and UL using the 
        // "interval estimation" technique proposed in Hu paper;
        mcf( _delta, _epsilon1, 1); // flag=1;

        // (c) now, if anything goes wrong for some pathological testcase, 
        // have a brutal exit; this will require debugging;
        counter++;
        if ( counter >= 512) {
            printf("\nError:  Binary search of MCMCF took more than 512 iterations.");
            printf("\n        This is an unusual testcase or the code has a bug.\n");
            free_topology();
            exit(1);
        }   
    }
    
    //printf("\nLL=%lf, UL=%lf", LL, UL);
    //printf("\nFinal latency L=%lf\n", UL);
    return 1;
}

double MCF::mcf( double delta, double epsilon, int flag)
{
    // flag:
    // 0 -- max concurrent flow;
    // 1 -- min-cost max concurrent flow;

    int i,j;
    int iter=0; // phase counter: number of iterations of the big main loop;
    double lambda=1; // result to be returned;
    double D=1, alpha=1, dual=1;
    // used to find the amount of flow pushed in each step;
    double usable_amount_cap, usable_amount_latency, routed_amount;
    // for tracking gap between lambda and dual;
    double gap=0.0, old_gap=0.0, old_old_gap=0.0;


    // () initialization of primal variables (i.e., flows thru all edges)
    // and dual valiables PHI_d, Y_e and "length function" l(e)
    // of all edges; also resets left_demand to demand for all commodities
    // as well as _total_latency;
    initialize( delta, flag);
    _rd = 1;
    for ( i = 0; i < no_node; i++) {
        nodes[i].dij_visited = 0;
        nodes[i].dij_updated = 0;
        nodes[i].min_visited = 0;
    }
    

    // () the big loop, each run of this loop is a phase; each phase
    // has |S| iterations;
    while (1) {

        // () in every phase we start with the demand d_j for every commodity;
        for ( j = 0; j < no_commodity; j++) {
            _commodities[j].left_demand = _commodities[j].demand;
        }
    

        // () next there are |S| iterations, one for each node that is a 
        // source for at least a commodity;
        for ( i = 0; i < no_node; i++) {
            if ( nodes[i].no_comm) { // if this node is source of "r" _commodities;
                
                int commodities_left = nodes[i].no_comm;
                int *dest_flag = (int*)malloc((no_node)*sizeof(int));
                if ( dest_flag == NULL) {
                    printf("\nError:  Unable to malloc <mcf>.\n"); exit(1);
                }
                memset((void*)dest_flag,0,(no_node)*sizeof(int));
                // dest_flag is set "1" for nodes that are destinations of _commodities;
                for ( j = 0; j < nodes[i].no_comm; j++) {
                    dest_flag[_commodities[nodes[i].comms[j]].dest] = 1;
                }


                // while there are left commodities to be routed for this node; 
                // there are a number of steps for current iteration;
                int step_count = 0;
                while ( commodities_left) {
                    step_count ++;

                    // () compute shortest PATHS tree, where edges have "length(e)";
                    // of all paths from this sink to all its destinations;
                    //print_network_demands( true); // debug;
                    shortest_paths( nodes[i].id, commodities_left, dest_flag);
                    
                    // () reset left_capacities of edges in the tree to the
                    // initial capacities; u'(e) = u(e), any e in tree;
                    reset_left_capacities_in_tree( nodes[i].id);

                    // () route "f = d(c_q)" units of flow of a given commodity
                    // and update the flow of each edge: f_e = f_e + f, along its 
                    // shortest path;
                    bool flow_has_been_routed = false;
                    for ( j = 0; j < nodes[i].no_comm; j++) {

                        // the amount of commodity c_q that has not been routed yet
                        // at step "s";
                        double left_demand = _commodities[nodes[i].comms[j]].left_demand;

                        if ( left_demand > 1e-3) {
                            flow_has_been_routed = true;
                            //print_backward_shortest_path(_commodities[nodes[i].comms[j]].dest);


                            // available flow amount from bottleneck-edge of shortest path;
                            // this "c" represents the available minimum capacity of the
                            // edges on shortest path of this commodity;
                            usable_amount_cap = min_capacity_this_commodity(
                                                                            _commodities[nodes[i].comms[j]].dest);

                            // available flow amount from latency constraint
                            if ( flag == 0) { // 0: "max concurrent flow"
                                usable_amount_latency = 1000000.0; // inf;
                            } else { // 1: "min-cost max concurrent flow"
                                // this is L/c(P), where c(P) is is the cost of sending 
                                // one unit of flow along the shortest path:
                                // Sum_{e in P}{D(e)}, where D(e) is latency of each edge;
                                usable_amount_latency = check_latency_constraint(
                                                                                 _commodities[nodes[i].comms[j]].dest);
                            }
                            
                            // flow amount to be routed at step "s": f_{i,j,s}^{c_q};
                            routed_amount = minimum(
                                                    usable_amount_cap, left_demand, usable_amount_latency);

                            // update every "_temp_edge_flow" - from dest backward to src
                            // will be added routed_amount; also update left_capacities
                            // of edges along the shortest path of this commodity;
                            route_flow( _commodities[nodes[i].comms[j]].dest,
                                        routed_amount, nodes[i].comms[j]);

                            // update commodity amounts to be routed still (i.e., are left);
                            _commodities[nodes[i].comms[j]].left_demand -= routed_amount;

                            if ( _commodities[nodes[i].comms[j]].left_demand <= 1e-3) {
                                // this commodity is done, set its destination flag to 0;
                                commodities_left --;
                                dest_flag[_commodities[nodes[i].comms[j]].dest] = 0;
                            }
                        }
                    }//for ( j = 0; j < nodes[i].no_comm; j++)

                    // () update dual variables: Y_e, phi_latency (or PHI_d), 
                    // length(e);
                    update_dual_variables( nodes[i].id, epsilon, flag);

                    _rd++;
                    if ( !flow_has_been_routed) break;
                }//while ( commodities_left)


                free( dest_flag);

            }//if ( nodes[i].no_comm)
        }//for ( i = 0; i < no_node; i++)


        // () increment phase counter; a phase is an iteration of the big main loop;
        iter++;
        // additional stopping criterion;
        if ( iter >= _scale) break;
        //if ( iter >= 80) break;

        // () compute dual and lambda and keep track of the gap between them;
        // -- compute dual=D/alpha;
        D = compute_D();
        alpha = compute_alpha();
        dual = D / alpha;

        // -- compute lambda;
        // Note1: the original code of Hu computed lambda differently;
        // this is not in fact lambda in the sense of Karakostas paper,
        // but rather an "artificial" variable to make easier its tracking
        // towards a value of 1;
        //lambda = L / (_total_latency/iter);
        // Note2: I now compute it as: lambda=MIN(actual flow/demand) among all commodities;
        lambda = compute_lambda();
        lambda /= iter; // consider effect of scaling;
        //printf("\n Lambda=%.8f, Dual=D/alpha=%.8f, D=%.8f",lambda,dual,D);
        
        // -- keep track of gap;
        old_old_gap = old_gap;
        old_gap = gap;
        gap = dual/lambda - 1;

         
        // () this implements the "interval estimation"; see Theorem 3 of Hu paper;
        if ( flag == 1) {
            double UL1 = UL, LL1 = LL;
            double d = dual;
            //if (d < 1) d = 1;
            double s1 = (_latency_max - L)/(_lambda_max - d);
            if ( s1 > 0 && ( _s < 0 || _s > s1)) { 
                _s = s1; 
            }
            if ( _s > 0) {
                if ( lambda < 1) {
                    UL1 = L + (1 - lambda) * _s;
                    if ( UL1 < UL) UL = UL1;
                }
                if ( dual > 1) {
                    LL1 = L - (dual - 1) * _s;
                    if ( LL1 > LL) LL = LL1;
                }
            }
            if ( lambda > 1) { UL = L; }
            if ( dual < 1) { LL = L; }
            if ( (UL-LL < 0) || (UL/LL - 1) < 0.01) { break; }
            if ( D >= 1) { break; }
        } else { // 0
            // for "max commodity flow" case, the stopping criterion is 
            // "D>=1"; see Karakostas paper;
            // Note1: original code of Hu used "dual/lambda-1<epsilon1";
            if ( D >= 1) { break; }
        }
    
    }//while (1)


    // () scale down the final flows so that the solution is feasible 
    // (that is, capacities are met);
    scale_down_flows( iter);
    // also, record final latency, which must consider scaling too;
    _total_latency = _total_latency / iter;

    
    // () entertain user;
    printf("\nlambda = %lf, dual = %lf, [%lf, %lf], L=%lf, iter=%d",
           lambda, D/alpha, LL, UL, L, iter);

    return lambda;
}

////////////////////////////////////////////////////////////////////////////////
//
// MCF Dijkstra
//
////////////////////////////////////////////////////////////////////////////////

void MCF::shortest_paths( int s, int num_commodities, int *dest_flag) 
{
    // implements Dijkstra's all paths shortest path algorithm;
    // num_commodities is the number of commodities that still need
    // routing for this source; 

    int num_commodities_to_process = num_commodities;
    PQDATUM wf, wf1; // WAVEFRONTS;

    PQUEUE pq;
    int *pos = (int *)malloc(sizeof(int) * (no_node));
    if ( pos == NULL) {
        printf("\nError: Unable to malloc <shortest_path>.\n"); exit (1);
    }
    
    pq.pqinit( 400); // 400 is just a constant;
    // reset dist of all nodes;
    for ( int i = 0; i < no_node; i++) {
        nodes[i].dist = DBL_MAX;
    }
    // source "s" resets;
    nodes[s].pre = -1;
    nodes[s].pre_edge = -1;
    nodes[s].dist = 0.0;

    wf.set_node( s); // sourse "s";
    wf.set_dist( 0.0);

    pq.pqinsert( wf, pos);

    while ( !pq.pqempty()) {
        int v, w;
        
        // retreive the shortest non-visited node;
        pq.pqremove( &wf1, pos);
        v = wf1.node();
        if ( dest_flag[v]) num_commodities_to_process--;
        // break when all shortest paths to all destinations from source "s"
        // have been found;
        if ( num_commodities_to_process <= 0) break;
        
        nodes[v].dij_visited = _rd;
        for ( int i = 0; i < nodes[v].no_edge; i++) {
            w = edges[nodes[v].edges[i]].dest;
            if ( nodes[w].dij_visited != _rd)
                if ( nodes[w].dij_updated != _rd ||
                     nodes[w].dist > wf1.dist() + edges[nodes[v].edges[i]].length) {
                    nodes[w].pre = v;
                    nodes[w].pre_edge = nodes[v].edges[i];
                    nodes[w].dist = wf1.dist() + edges[nodes[v].edges[i]].length;
                    wf.set_node( w);
                    wf.set_dist( nodes[w].dist);
                    if (nodes[w].dij_updated != _rd) {
                        pq.pqinsert( wf, pos);
                    } else {
                        pq.pqdeckey( wf, pos);
                    }
                    nodes[w].dij_updated = _rd;
                }
        }
    }
    pq.pqfree( pos); // clean up;
}

////////////////////////////////////////////////////////////////////////////////
//
// MCF network graph
//
////////////////////////////////////////////////////////////////////////////////

bool MCF::build_network_from_host_application()
{
    // used from inside the host application that hosts also the floorplanner
    // and the VNOC1 NoC simulator;
    // you should implement this based on how your host application looks like;
    // the idea is to populate the MCF object similarly to how I do it inside
    // read_network_topology_and_demands();

    return true;
}

bool MCF::build_network_from_file( double latency_limit, double rate)
{
    // rate is the demand coefficient (injection rate): 0.05, 0.1, 0.15, 0.2, 0.25;
    // latency_limit and rate are not used;

    FILE *fp; // file pointer for network file;

    // (1) import the network topology and the demands;
    if (( fp = fopen( _network_filename, "r"))) {
        read_network_topology_and_demands( fp);
    } else {
        printf("Error:  Can not open file: %s\n", _network_filename); exit(1);
    }

    // (2) cleanup;
    fclose( fp);

    // (3) one time initialization of parameters (of those not set by
    // user via command line arguments);
    init_param();

    return true;
}

void MCF::read_network_topology_and_demands( FILE *fp, double rate) 
{
    // Note: I assume that network file is correct; I do not do sanity 
    // checks for the time being;
    // I "made-up" this format for easy parsing; you may want to change
    // to fit your application; example of .network format:
    //
    // 8 <-- num of nodes
    // 0 100 300 <-- node id, (x,y) location in um
    // 1 100 100
    // 2 200 300
    // 3 200 100
    // 4 300 300
    // 5 300 100
    // 6 400 300
    // 7 400 100
    // 12 <-- num of edges
    // 0 0 2 10.0 2.00 <-- id, src, des, capacity, delay
    // 1 2 4 10.0 2.00 
    // 2 3 2 10.0 6.00 
    // 3 1 3 10.0 2.00 
    // 4 3 5 10.0 2.00
    // 5 2 3 10.0 6.00 
    // 6 4 2 10.0 2.00
    // 7 5 3 10.0 2.00
    // 8 5 4 10.0 6.00 
    // 9 4 5 10.0 6.00 
    // 10 4 6 10.0 2.00
    // 11 5 7 10.0 2.00
    // 2 <-- num of demands (commodities)
    // 0 0 7 0.577004 <-- id src des amount
    // 1 1 6 1.777268

    int id, x, y, src, dest;
    double delay, capacity;

    // (1) nodes
    fscanf(fp,"%d", &no_node);
    nodes = (NODE*)malloc(sizeof(NODE) * (no_node));
    if ( nodes == NULL) {
        printf("\nError: Unable to malloc <nodes>.\n"); exit(1);
    }
    for ( int i = 0; i < no_node; i++) {
        fscanf(fp, "%d %d %d", &id, &x, &y);
        nodes[i].id = id;
        nodes[i].x = x;
        nodes[i].y = y;
        nodes[i].pre = -1;
        nodes[i].dist = DBL_MAX;
        nodes[i].no_comm = 0;
        nodes[i].comms = NULL;
        nodes[i].no_edge = 0;
        nodes[i].dij_visited = 0;
        nodes[i].dij_updated = 0;
        nodes[i].min_visited = 0;

        // here we work with a fixed pre-allocation; not optimal; we should
        // allocate only as much as needed; also in this way we have to make
        // sure there will be no nodes with a bigger degree than MAX_DEGREE=40;
        // TO DO: this must be fixed as it's ugly programming;
        nodes[i].edges = (int *)malloc(sizeof(int) * MAX_DEGREE);
        if ( nodes[i].edges == NULL) {
            printf("\nError: Unable to malloc <nodes.edges>.\n"); exit(1);
        }
    }

    // (2) edges
    fscanf(fp,"%d", &no_edge);
    edges = (EDGE *)malloc(sizeof(EDGE) * (no_edge));
    if ( edges == NULL) {
        printf("\nError: Unable to malloc <edges>.\n"); exit(1);
    }
    _temp_edge_flow = (double*)malloc(sizeof(double) * (no_edge));
    if ( _temp_edge_flow == NULL) {
        printf("\nError: Unable to malloc <_temp_edge_flow>.\n"); exit(1);
    }
    for ( int i = 0; i < no_edge; i++) {
        fscanf(fp, "%d %d %d %lf %lf",&id, &src,&dest, &capacity, &delay);
        edges[i].id = id;
        edges[i].src = src;
        edges[i].dest = dest;
        edges[i].capacity = capacity;
        edges[i].left_capacity = capacity;
        edges[i].latency = delay;
        edges[i].length = 0.0;

        edges[i].flow = 0.0;
        edges[i]._flows = NULL;
    }

    // (3) record adjacent edges for each node;
    for ( int i = 0; i < no_edge; i++) {
        int index = edges[i].src;
        nodes[index].edges[nodes[index].no_edge] = edges[i].id;
        nodes[index].no_edge++;
    }

    // (4) read demands/commodities;
    double amount;
    fscanf(fp,"%d", &no_commodity);
    _commodities = (COMMODITY *)malloc(sizeof(COMMODITY) * (no_commodity));
    if ( _commodities == NULL) {
        printf("\nError: Unable to malloc <_commodities>.\n"); exit(1);
    }
    for ( int i = 0; i < no_commodity; i++) {
        fscanf(fp,"%d %d %d %lf", &id, &src, &dest, &amount);
        _commodities[i].id = id;
        _commodities[i].src = src;
        _commodities[i].dest = dest;
        _commodities[i].demand = amount * rate; // rate = 1 by default;
        _commodities[i].left_demand = amount;

        if (nodes[src].comms == NULL) {
            nodes[src].comms = (int *)malloc(sizeof(int) * no_node);
            if ( nodes[src].comms == NULL) {
                printf("\nError: Unable to malloc <nodes[src].comms>.\n"); exit(1);
            }
        }
        nodes[src].comms[nodes[src].no_comm] = i;
        nodes[src].no_comm++;
    }

    // (5) reset;
    for ( int i = 0; i < no_edge; i++) {
        // Note1: I had to delay this allocation because no_commodity had 
        // not been set yet;
        edges[i]._flows = (double *)malloc(sizeof(double) * (no_commodity));
        if ( edges[i]._flows == NULL) {
            printf("\nError: Unable to malloc <edges._flows>.\n"); exit(1);
        }
        for ( int j = 0; j < no_commodity; j++) {
            edges[i]._flows[j] = 0.0;
        }
    }
    for ( int i = 0; i < no_node; i++) {
        // Note2: same as above;
        nodes[i]._preferred_path = (int *)malloc(sizeof(int) * no_commodity);
        if ( nodes[i]._preferred_path == NULL) {
            printf("\nError: Unable to malloc <_preferred_path>.\n"); exit(1);
        }
        for ( int j = 0; j < no_commodity; j++) {
            nodes[i]._preferred_path[j] = -1;           
        }
    }
    
    //print_network_demands(); exit(1); // debug;

    return;
}

void MCF::free_topology()
{
    int i=0;

    free( _commodities);

    for ( i = 0; i < no_edge; i++) {
        free( edges[i]._flows);
    }
    free( edges);
    free( _temp_edge_flow);

    for ( i = 0; i < no_node; i++) {
        free( nodes[i].comms);
        free( nodes[i].edges);
    }
    free( nodes);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//
// debug utils;
//
////////////////////////////////////////////////////////////////////////////////

void MCF::print_network_demands( bool print_only_edges)
{
    printf("\n\nNetwork and demands:");
    printf("\nNodes %d",no_node);
    if ( print_only_edges == false) {
        for ( int i = 0; i < no_node; i++) {
            printf("\n %d  (%d %d)", nodes[i].id, nodes[i].x, nodes[i].y);
            printf("  num_commodities=%d dist=%lf",nodes[i].no_comm,nodes[i].dist);
            printf("\n     ");
            for ( int k = 0; k < nodes[i].no_edge; k++) {
                printf(" %d", nodes[i].edges[k]);
            }
        }
    }
    printf("\nEdges %d",no_edge);
    for ( int i = 0; i < no_edge; i++) {
        //printf("\n %d %d -> %d  cap: %.2lf  Y_e: %.2lf  len: %.2lf  flow: %.2lf  breakdown:",
        //  edges[i].id, edges[i].src, edges[i].dest,
        //  edges[i].capacity, edges[i]._Y_e,
        //  edges[i].length, edges[i].flow);
        printf("\n %d %d -> %d  cap: %.2lf  flow: %.2lf  breakdown:",
               edges[i].id, edges[i].src, edges[i].dest,
               edges[i].capacity, edges[i].flow);
        for ( int j = 0; j < no_commodity; j++) {
            printf(" %.2lf", edges[i]._flows[ j]);
        }
    }
    if ( print_only_edges == false) {
        printf("\nDemands/commodities %d",no_commodity);
        for ( int i = 0; i < no_commodity; i++) {
            printf("\n %d  %d -> %d demand=%lf  portion_unsatisfied=%lf", _commodities[i].id,
                   _commodities[i].src, _commodities[i].dest,
                   _commodities[i].demand, // amount * rate
                   _commodities[i].left_demand); // amount
        }
    }
    printf("\n");
}

void MCF::print_routing_paths()
{
    // call only after a call of do_randomized_rounding();
    printf("\nRandomized rounded paths:");

    for ( int i = 0; i < no_commodity; i++) {
        printf("\nCommodity %d: %d -> %d: ", i, 
               _commodities[i].src, _commodities[i].dest);

        int src_id = _commodities[i].src;
        while ( src_id != _commodities[i].dest) {
            printf(" %d", src_id);
            src_id = nodes[src_id]._preferred_path[i];
        }
        printf(" %d", src_id); // dest;
    }
    printf("\n");
}

void MCF::print_backward_shortest_path( int dest)
{
    // debug only;
    int t = dest;
    printf("\n");
    while ( nodes[t].pre != -1) {
        printf(" %d ->", t);
        t = nodes[t].pre;
    } printf(" %d ", t);
}

////////////////////////////////////////////////////////////////////////////////
//
// MCF randomized rounding;
//
////////////////////////////////////////////////////////////////////////////////

bool MCF::do_randomized_rounding()
{
    // after mcf_solver finds a fractional flow solution, we do a 
    // randomized rounding to set only one path for each commodity;
    // otherwise the commodities would traverse multiple paths that
    // would translate in path splitting for packets which would require
    // router architecture modification too much and re-ordering of packets
    // at destination;

    for ( int i = 0; i < no_commodity; i++) {
        // () for each commodity we start from its source and traverse
        // downstream nodes; at each step we pick up the node that has
        // the largest fraction of this commodity as the preferred path;
        // record this preferred path in the preferred_path array;

        int src_id = _commodities[i].src;
        while ( src_id != _commodities[i].dest) {
            // recursively travel from src to dest searching for maximum
            // fractional flow;
                
            int id_max_flow_fraction = -1;
            double max_flow_fraction = 0.0;
            for ( int k = 0; k < nodes[ src_id].no_edge; k++) {
                // for each adjacent edge look at the commodity index "i",
                // and seek the edge id with maximum flow fraction;

                int edge_id = nodes[ src_id].edges[k];
                if ( max_flow_fraction < edges[edge_id]._flows[ i]) {
                    max_flow_fraction = edges[edge_id]._flows[ i];
                    id_max_flow_fraction = edge_id;
                }
            }
            assert(id_max_flow_fraction >= 0 & id_max_flow_fraction < no_edge);

            // () record the neighbor node id as the downstream node of the
            // preferred path for this commodity; that is, along the edge with
            // max fraction of flow from current node for this commodity;
            nodes[src_id]._preferred_path[i] = edges[id_max_flow_fraction].dest;

            // prepare for next iter;
            src_id = nodes[src_id]._preferred_path[i];
        }
    }

    return true;
}
}
