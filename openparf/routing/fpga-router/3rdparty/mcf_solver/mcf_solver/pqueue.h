#ifndef _PQUEUE_H_
#define _PQUEUE_H_

#include <stdio.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//
// PQUEUE
//
////////////////////////////////////////////////////////////////////////////////

class PQDATUM
{
 private:
    int _node;
    double _dist;
 public:
    PQDATUM() { _node = -1; _dist = -1; }
    ~PQDATUM() {}

    int node() { return _node; }
    double dist() { return _dist; }
    void set_node(int node) { _node = node; }
    void set_dist(double dist) { _dist = dist; }
};

class PQUEUE
{
 private:
    int _size, _avail, _step;
    PQDATUM *_d;
 public:
    PQUEUE() { _size = -1; _avail = -1; _step = -1; }
    ~PQUEUE() {}

    int size() { return _size; }
    int avail() { return _avail; }
    int step() { return _step; }
    PQDATUM *d() { return _d; }
    PQUEUE *pqinit(int n);
    int pqinsert( PQDATUM a_d, int *pos);
    PQDATUM *pqremove( PQDATUM *a_d, int *pos);
    int pqdeckey( PQDATUM a_d, int *pos);
    PQDATUM *pqpeek( PQDATUM *a_d);
    double get_distance( PQDATUM d) { return ( d.dist()); }
    int pqempty() { return ( _size == 1); }
    void pqfree( int *pos) {
        free( _d);
        free( pos);
    }
};

#endif
