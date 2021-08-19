#include <stdio.h>
#include <stdlib.h>
#include <float.h>
// #define NDEBUG // uncomment to disable asserts
#include <cassert>
#include "pqueue.h"

using namespace std;

#define assertm(exp, msg) assert(((void)msg, exp))

////////////////////////////////////////////////////////////////////////////////
//
// PQUEUE
// _d[1] is the datum with the smallest _dist
// guarantee: _d[i/2] has smaller _dist than _d[i]; integer division; that is _d[1] <= _d[2], _d[1] <= _d[3]
// guarantee: _positions[_d[i].node()] == i; that is, _positions is a valid backpointer
// note: _d[0] is empty to make integer math simple
// note: node ids are assumed to go from 0 to __size -1
// note: datum points are stored at _d[1] ... _d[size]
//
////////////////////////////////////////////////////////////////////////////////

bool PQUEUE::pqinit( int n) 
{
    assertm(n > 0, "positive number of entries?");
    assertm(_d == nullptr, "init with non-null data");

    // initialize the queue;
    _d = new PQDATUM[n+1]; // note, we never use d[0]

    _positions = new int[n]; // position values will be 1 ... n
    for (int nid = 0; nid < n; nid++)
        _positions[nid] = -1;

    assertm(_d != nullptr, "unable to allocate space");
    
    _avail = n;
    _size = 0;

    return _d != nullptr && _positions != nullptr;
}


bool PQUEUE::pqinsert( PQDATUM const datum)
{
    PQDATUM *tmp;
    int newpos;

    assertm(_d != nullptr, "call pqinit first");
    assertm(_size + 1 <= _avail, "not enough memory");
    assertm(datum.node() >= 0 && datum.node() <= _avail, "datum node id is not legit");

    newpos = ++_size;
    while (newpos > 1 && _d[newpos / 2].dist() > datum.dist()) {
        _d[newpos] = _d[newpos / 2];
        _positions[ _d[newpos].node()] = newpos;
        newpos /= 2;
    }

    // deep copy
    *(&(_d[newpos])) = datum;
    _positions[ _d[newpos].node()] = newpos;
    return true;
} 

// removes datum with smallest _distance
bool PQUEUE::pqremove( PQDATUM *answer)
{
    PQDATUM tmp;
    int curr = 1, next;

    assertm(_size > 0 && _d != nullptr, "no elements or not pqinit?");

    *answer = _d[curr];
    _positions[_d[curr].node()] = -1; // reset
    _size--;

    if (_size == 0) return true;

    // bubble up the next smallest into _d[1] and remove the old element at _d[_size + 1]
    tmp = _d[ _size+1];
    while (curr <= _size / 2) {
        next = 2 * curr;
        if ( next < _size && _d[next].dist() > _d[next + 1].dist()) {
            next++;
        }
        if ( _d[next].dist() >= tmp.dist()) {
            break;
        }
        _d[curr] = _d[next];
        _positions[ _d[curr].node()] = curr;
        curr = next;
    }

    // deep copy
    *(&(_d[curr])) = tmp;
    _positions[ _d[curr].node()] = curr;
    return true;
} 

bool PQUEUE::pqdecrease( int node, double new_distance)
{
    assertm(_size > 0 && _d != nullptr, "no elements or not init?");
    assertm(node >= 0 && node < _avail, "node id not in positions");
    assertm(_d[_positions[node]].node() == node, "node at position is not the same");
    assertm(_d[_positions[node]].dist() > new_distance, "distance not decreasing");

    int curr = _positions[node];
    _d[curr].set_dist(new_distance);

    PQDATUM tmp = _d[curr];

    // since distance has decreased, this node can only move up in the priority queue
    while ( curr > 1 && _d[curr / 2].dist() > new_distance) {
        _d[curr] = _d[curr / 2];
        _positions[ _d[curr].node()] = curr;
        curr /= 2;
    }
    *(&(_d[curr])) = tmp;
    _positions[ _d[curr].node()] = curr;
    return true;
}

bool PQUEUE::pqpeek( PQDATUM *answer)
{
    assertm(_size > 0 && _d != nullptr, "no elements or not init?");
    assertm(answer != nullptr, "answer can't be null");

    *answer = _d[1];
    return true;
}

double PQUEUE::pqpeeklength(int nodeid)
{
    assertm(nodeid >= 0 && nodeid < _avail, "nodeid outside scope");
    int pos = _positions[nodeid];
    assertm(pos >= 1 && pos <= _size, "pos outside scope");
    return _d[pos].dist();
}