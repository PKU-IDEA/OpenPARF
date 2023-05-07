#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "pqueue.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//
// PQUEUE
//
////////////////////////////////////////////////////////////////////////////////

PQUEUE *PQUEUE::pqinit( int n) 
{
    // initialize the queue;
    _d = (PQDATUM *)malloc(sizeof(PQDATUM) * n);
    if ( _d == NULL) {
        printf("\nError: Unable to realloc <pqinit>.\n");
        exit (1);
    }
    
    _avail = _step = n;
    _size = 1;
    return ( this); // return a pointer to itself;
}

int PQUEUE::pqinsert( PQDATUM a_d, int *pos)
{
    // insert an item into the queue;
    // return 1 if item was inserted; 0 otherwise;

    PQDATUM *tmp;
    int i, newsize;

    if ( _size == -1) return 0; // pqueue was not initialized first;
    
    // (1) allocate more memory if necessary;
    if ( _size >= _avail) {
        newsize = _size + _step;
        tmp = (PQDATUM *)realloc( _d, sizeof(PQDATUM) * newsize);
        if ( tmp == NULL) {
            printf("\nError: Unable to realloc <pqinsert>.\n");
            exit (1);
            //return 0;
        }
        _d = tmp; // redundant;
        _avail = newsize;       
    }

    // (2) insert item;
    i = _size++;
    while ( i > 1 && get_distance( _d[i / 2]) > get_distance( a_d)) {
        _d[i] = _d[i / 2];
        pos[ _d[i].node()] = i;
        i /= 2;
    }
    _d[i] = a_d;
    pos[ _d[i].node()] = i;
    return 1;   
} 

PQDATUM *PQUEUE::pqremove( PQDATUM *a_d, int *pos)
{
    // remove the highest-ranking item from the queue;
    // a_d: pointer to the PQDATUM variable that will hold the
    // datum corresponding to the queue item removed;
    // return value:
    // >= 0  an item has been removed. The variable that d points
    //       to now contains the datum associated with the item in question;
    // -1    no item could be removed. Either the queue pointer
    //       provided was NULL, or the queue was empty. The chunk
    //       of memory that d points to has not been modified.

    PQDATUM tmp;
    int i = 1, j;

    if ( _size == -1 || _size == 1) return NULL;

    *a_d = _d[1];
    tmp = _d[ --_size];
    while (i <= _size / 2) {
        j = 2 * i;
        if ( j < _size && get_distance(_d[j]) > get_distance(_d[j + 1])) {
            j++;
        }
        if ( get_distance(_d[j]) >= get_distance(tmp)) {
            break;
        }
        _d[i] = _d[j];
        pos[ _d[i].node()] = i;
        i = j;
    }
    _d[i] = tmp;
    pos[ _d[i].node()] = i;
    return a_d; 
} 

int PQUEUE::pqdeckey( PQDATUM a_d, int *pos)
{
    int i = 0;

    if ( _size == -1) return 0; // pqueue was not initialized first;

    i = pos[ a_d.node()];
    if ( _d[i].node() != a_d.node())
        printf("wrong\n");
    while ( i > 1 && get_distance(_d[i / 2]) > get_distance(a_d)) {
        _d[i] = _d[i / 2];
        pos[ _d[i].node()] = i;
        i /= 2;
    }
    _d[i] = a_d;
    pos[ _d[i].node()] = i;
    return 1;
}

PQDATUM *PQUEUE::pqpeek( PQDATUM *a_d)
{
    // access highest-ranking item without removing it;
    // a_d: pointer to the PQDATUM variable that will hold the
    // datum corresponding to the highest-ranking item;
    // return value:
    // >= 0  Success. The variable that d points to now contains
    //       the datum associated with the highest-ranking item.
    // -1    Failure. Either the queue pointer provided was NULL,
    //       or the queue was empty. The chunk of memory that d
    //       points to has not been modified.

    if ( _size == -1 || _size == 1) return NULL;

    *a_d = _d[1];
    return a_d;
}
