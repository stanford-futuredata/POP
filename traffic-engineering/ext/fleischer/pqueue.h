#ifndef _PQUEUE_H_
#define _PQUEUE_H_

#include <iostream>

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

    int node() const { return _node; }
    double dist() const { return _dist; }
    void set_node(int node) { _node = node; }
    void set_dist(double dist) { _dist = dist; }
};

class PQUEUE
{
 private:
    int _size, _avail, _step;
    PQDATUM *_d;
    int* _positions;
 public:
     PQUEUE() : _size(0), _avail(-1), _d(nullptr), _positions(nullptr) {}
     PQUEUE(int n) : PQUEUE() { pqinit(n); }
     ~PQUEUE() { 
         cout << "here" << endl;  
         if (_d != nullptr) { delete[] _positions; delete[] _d; }
         cout << "and here?" << endl;
     }

    int size() { return _size; }
    int avail() { return _avail; }
    int step() { return _step; }
    // cannot change contents outside of class
    const PQDATUM *d() { return _d; }
    const int* positions() { return _positions; }

    // init
    bool pqinit(int n);

    // main
    bool pqinsert( const PQDATUM datum); // pos is a back-pointer from node_id to position in PQ
    bool pqremove( PQDATUM *answer);
    bool pqdecrease( int node, double new_distance);
    bool pqpeek( PQDATUM *answer);
    double pqpeeklength(int nodeid);


    // conveninece methods
    double get_key( PQDATUM d) { return d.dist(); }
    bool pqempty() { return ( _size == 0); }
};

#endif
