#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include "pqueue.h"
#include <sys/time.h>

// #define NDEBUG // uncomment to disable asserts
#include <cassert>
using namespace std;
#define assertm(exp, msg) assert(((void)msg, exp))

#define MAXLINELEN 100

class Node
{
public:
    int id; // a number between 0 and N-1
    map<int, int> out_nbrs; // nbr id -> edge id
    map<int, int> in_nbrs; // nbr id -> edge_id
    list<int> from_commods; // commod_ids which begin at this node

    // dirtiable state for dijkstra
    bool visited;
    double shortest_path_length;
    int pre_id; // back-pointer, =-1 if not set
};

class Edge
{
public:
    int id; // a number between 0 and M-1
    int u, v;
    double cap;

    // dirtiable state with path length
    double length;
    map<int, double> flow;
    double total_flow;
};

class Commodity
{
public:
    int id;
    int n_source, n_target;
    double demand;

    double flow_alloc;

    friend std::ostream& operator<<(std::ostream& out, const Commodity& data)
    {
        out << "C" << data.id << 
            "(" << data.n_source << "->" << data.n_target << "): " 
            << data.demand << "; alloc= " << data.flow_alloc;
    }
};

class HeapItem
{
public:
    int node_id;
    double path_length;
    HeapItem(int n, double d) : node_id(n), path_length(d) {}
    HeapItem() : node_id(-1), path_length(DBL_MAX) {}
};

struct compare_heap_items {
    bool operator()(const HeapItem* h1, const HeapItem* h2) const {
        return h1->path_length > h2->path_length;
    }
};

string print_vector(vector<int> incoming)
{
    stringstream retval;
    retval << "[";
    for (vector<int>::iterator vii = incoming.begin(); vii != incoming.end(); vii++)
    {
        if (vii != incoming.begin()) retval << ",";
        retval << *vii;
    }
    retval << "]";

    return retval.str();
}

const string WHITESPACE = " \n\r\t\f\v";

string ltrim(const string& s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == string::npos) ? "" : s.substr(start);
}

string rtrim(const string& s)
{
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == string::npos) ? "" : s.substr(0, end + 1);
}

string trim(const string& s)
{
    return rtrim(ltrim(s));
}


// global variables
Node* g_nodes;
Edge* g_edges;
Commodity* g_commods;
int g_num_nodes, g_num_edges, g_num_demands;
double g_epsilon = 0.1;
int g_num_dijkstra_calls = 0;
int g_num_shortest_path_calls = 0;
HeapItem* g_heapitems;
PQUEUE* g_pq;
int**** g_paths; // src_id -> target_id -> path_id -> [#edges=k, eid1 ... eidk]
int** g_num_paths; // src_id -> target_id -> num_paths
pair<int, double>** g_best_path;

void read_paths(ifstream& input)
{
    bool debug = false;
    bool found_path;

    do {
        string line;
        int from_node, to_node;
        string arrow;
        vector<vector<int>> paths;

        found_path = false;

        while (std::getline(input, line))
        {
            if (debug) cout << "read_path-- [" << line << "]" << endl;

            line = ltrim(rtrim(line));
            if (line.length() == 0)
            {
                if (found_path) break; // break at whitespace if path is found
                else continue; // else, skip
            }

            if (!found_path)
            {
                stringstream(line) >> from_node >> arrow >> to_node;
                found_path = true;
                if (debug) cout << from_node << arrow << to_node << endl;
            }
            else
            {
                bool add_path = true;
                vector<int> path_elements;
                int index_begin = 1, index_end = line.length() - 1, next_comma = 0;
                int prev = -1;
                while (next_comma != string::npos)
                {
                    next_comma = line.find(",", index_begin);
                    if (debug) cout << index_begin << " " << next_comma << " " << index_end << endl;
                    string s =
                        next_comma == string::npos ?
                        line.substr(index_begin, index_end - index_begin) :
                        line.substr(index_begin, next_comma - index_begin);
                    if (debug) cout << s << endl;

                    int val;
                    stringstream(s) >> val;
                    if (prev != -1) {
                      Node& pn = g_nodes[prev];
                      Edge& e = g_edges[pn.out_nbrs[val]];
                      if (e.cap == 0.0) {
                        add_path = false;
                      }
                    }
                    path_elements.push_back(val);
                    prev = val;

                    index_begin = next_comma + 1;
                }

                assertm(path_elements.size() > 0, "path_elements too small");
                assertm(path_elements.front() == from_node && path_elements.back() == to_node, "path should start and end appropriately");

                if (add_path) {
                  paths.push_back(path_elements);
                }
            }
        }

        g_num_paths[from_node][to_node] = paths.size();
        if (found_path && paths.size() > 0)
        {
            assertm(from_node <= g_num_nodes && from_node >= 0, "from_node not legit");
            assertm(to_node <= g_num_nodes && to_node >= 0, "to_node not legit");
            // assertm(paths.size() > 0, "paths too small");

            if (debug) cout << "F" << from_node << " T" << to_node << " #paths=" << paths.size() << endl;

            g_paths[from_node][to_node] = new int* [paths.size()];

            int path_id = 0;
            for (vector<vector<int>>::iterator vvii = paths.begin();
                vvii != paths.end();
                vvii++)
            {
                vector<int>& vp = *vvii;
                if (debug) cout << "\t" << print_vector(vp) << endl;
                int* path = new int[vp.size()];
                path[0] = vp.size() - 1;

                int curr = 1;
                int prev;
                for (vector<int>::iterator vii = vp.begin(); vii != vp.end(); vii++)
                {
                    if (vii != vp.begin())
                    {
                        Node& pn = g_nodes[prev];
                        Edge& e = g_edges[pn.out_nbrs[*vii]];
                        path[curr++] = e.id;
                    }
                    prev = *vii;
                }
                g_paths[from_node][to_node][path_id++] = path;
                assertm(curr == path[0] + 1, "must have pushed exactly these many edges");
            }
        }
    } while (found_path);
}

void ReadTopoAndDemands(char* fname, bool readPaths=false)
{
    bool debug = true;
    // Note: I assume that network file is correct; I do not do sanity 
    // checks for the time being;
    // I "made-up" this format for easy parsing; you may want to change
    // to fit your application; example of .network format:
    //
    // 4 <-- num of nodes
    // 0 100 300 <-- node id, (x,y) location in um
    // 1 100 100
    // 2 400 300
    // 3 400 100
    // 3 <-- num of edges
    // 0 0 2 10.0 2.00 <-- id, src, des, capacity, delay
    // 1 2 4 10.0 2.00 
    // 2 3 2 10.0 6.00 
    // 2 <-- num of demands (commodities)
    // 0 0 7 0.577004 <-- id src des amount
    // 1 1 6 1.777268
    if (debug) cout << "fname -- " << fname << endl;
    ifstream input(fname);
    string line;

    // (1) nodes
    input >> g_num_nodes;
    g_nodes = new Node[g_num_nodes];
    if (debug) cout << "--x #nodes=" << g_num_nodes << endl;
    // this portion is nothing but garbage
    for (int n_id = 0; n_id < g_num_nodes; n_id++)
    {
        int id, src, des;
        input >> id >> src >> des;
        g_nodes[n_id].id = n_id;
    }

    // (2) edges 
    input >> g_num_edges;
    g_edges = new Edge[g_num_edges];
    if (debug) cout << "--x #edges=" << g_num_edges << endl;

    for (int e_id = 0; e_id < g_num_edges; e_id++)
    {
        int i1, i2, i3;
        double d1, d2;
        input >> i1 >> i2 >> i3 >> d1 >> d2;

        g_edges[e_id].id = e_id;
        g_edges[e_id].u = i2;
        g_edges[e_id].v = i3;
        g_edges[e_id].cap = d1;
        g_edges[e_id].total_flow = 0;

        g_nodes[i2].out_nbrs[i3] = e_id; // .push_back(i3);
        g_nodes[i3].in_nbrs[i2] = e_id; // .push_back(i2);
    }

    // (3) demands
    input >> g_num_demands;
    g_commods = new Commodity[g_num_demands];
    if (debug) cout << "--x #demands=" << g_num_demands << endl;

    for (int c_id = 0; c_id < g_num_demands; c_id++)
    {
        int i1, i2, i3;
        double d1;
        input >> i1 >> i2 >> i3 >> d1;

        g_commods[c_id].id = c_id;
        g_commods[c_id].n_source = i2;
        g_commods[c_id].n_target = i3;
        g_commods[c_id].demand = d1;

        g_nodes[i2].from_commods.push_back(c_id);
    }

    // (4) paths
    if (readPaths)
    {
        g_num_paths = new int* [g_num_nodes];
        g_paths = new int*** [g_num_nodes];
        for (int n_id_i = 0; n_id_i < g_num_nodes; n_id_i++)
        {
            g_num_paths[n_id_i] = new int[g_num_nodes];
            g_paths[n_id_i] = new int** [g_num_nodes];
        }

        for (int n_id_i = 0; n_id_i < g_num_nodes; n_id_i++)
            for (int n_id_j = n_id_i + 1; n_id_j < g_num_nodes; n_id_j++)
            {
                g_num_paths[n_id_i][n_id_j] = 0;
                g_num_paths[n_id_j][n_id_i] = 0;
            }

        read_paths(input);
    }

    if (debug) cout << "--x done?" << endl;
    return;
}

// when restricted to just a few paths, we explicitly count out the cost of each path
// and remember that path_id
void run_simple_shortest_paths(int src_id, double* weights, int target_id = -1, bool debug=false)
{
    // debug = true;
    g_num_shortest_path_calls++;
    if (g_num_shortest_path_calls == 1)
    {
        g_best_path = new pair<int, double> * [g_num_nodes];
        for (int i = 0; i < g_num_nodes; i++)
            g_best_path[i] = new pair<int, double>[g_num_nodes];

        assertm(g_paths != nullptr && g_num_paths != nullptr, "input must have paths");
    }

    Node& source = g_nodes[src_id];
    for (int nid = 0; nid < g_num_nodes; nid++)
    {
        if (nid == src_id) continue;
        if (target_id != -1 && nid != target_id) continue;

        // finding path from src_id to nid
        int num_paths = g_num_paths[src_id][nid];
        double min_path_length = DBL_MAX;
        double min_path_id = -1;
        for (int path_id = 0; path_id < num_paths; path_id++)
        {
            double path_length = 0;
            int* path = g_paths[src_id][nid][path_id];
            int num_edges = path[0];
            assertm(num_edges > 0, "too few edges?");

            for (int edge_index = 1; edge_index <= num_edges; edge_index++)
            {
                int edge_id = path[edge_index];
                if (edge_id < 0 || edge_id > g_num_edges)
                {
                    cout << "bad edge" << endl;
                }
                assertm(!(edge_id < 0 || edge_id > g_num_edges), "bad edge");
                path_length += weights[path[edge_index]];
            }

            if (path_length < min_path_length)
            {
                min_path_length = path_length;
                min_path_id = path_id;
            }
        }

        g_best_path[src_id][nid] = pair<int, double>(min_path_id, min_path_length);
        if (debug)
            cout << "gssp: " << src_id << " -> " << nid <<
            " num_paths= " << num_paths <<
            " min_path id= " << min_path_id <<
            " len= " << min_path_length <<
            endl;

        assertm((min_path_id >= 0 && min_path_length >= 0) || num_paths == 0, "no path?");
    }
}

void run_dijkstra(int src_id, double* weights, int target_id = -1)
{
    g_num_dijkstra_calls++;

    bool debug = false;

    if (debug) cout << "--Dijkstra" << g_num_dijkstra_calls << " start for " << src_id << endl;

    if (g_num_dijkstra_calls == 1)
    {
        // shared global fleischer across dijkstra runs
        g_pq = new PQUEUE(g_num_nodes);
    }

    // set-up
    assertm(g_pq != nullptr && g_pq->size() == 0 && g_pq->avail() >= g_num_nodes, "g_pq is not clean");

    PQDATUM item;
    for (int nid = 0; nid < g_num_nodes; nid++)
    {
        Node& n = g_nodes[nid];
        n.visited = false;
        
        item.set_node(nid);
        item.set_dist(nid == src_id ? 0 : DBL_MAX);
        assertm(g_pq->pqinsert(item), "cannot insert?");
    }

    Node* target = (target_id == -1) ? nullptr : &(g_nodes[target_id]);

    double max_path_length = 0, min_path_length = DBL_MAX;
    while (g_pq->size() > 0 && (target == nullptr || !target->visited))
    {
        assertm(g_pq->pqremove(&item), "cannot remove?");

        if (debug) cout << "--x visiting node:" << item.node() << " pl: " << item.dist() << endl;
        Node& node = g_nodes[item.node()];
        assertm(!node.visited, "can't visit again");

        node.visited = true;
        node.shortest_path_length = item.dist();
        if(max_path_length < node.shortest_path_length)
            max_path_length = node.shortest_path_length;

        if (node.id != src_id && min_path_length > node.shortest_path_length)
            min_path_length = node.shortest_path_length;

        map<int, int>::iterator miii;
        map<int, int>& src_out_nbrs = node.out_nbrs;
        for (miii = src_out_nbrs.begin(); miii != src_out_nbrs.end(); miii++)
        {
            Node& other = g_nodes[(*miii).first];
            if (other.visited) continue; // node already visited

            int edge_id = (*miii).second;
            double new_path_length = node.shortest_path_length + weights[edge_id];
            if (g_pq->pqpeeklength(other.id) > new_path_length)
            {
                g_nodes[other.id].pre_id = node.id;
                g_pq->pqdecrease(other.id, new_path_length);
            }

            if (debug) cout << "--x other: " << other.id << " pl: " << new_path_length << endl;
        }
    }

    if (target != nullptr)
    {
        // drain the priority queue off any other nodes
        while (g_pq->size() > 0)
            g_pq->pqremove(&item);
    }

    if (debug)
        cout << "--z dijkstra APSP from src= " << src_id 
        << " path_lengths in [" << min_path_length << ", " << max_path_length << "]" 
        << endl;
    if (debug) {
        string whatever;
        cin >> whatever;
    }
}


double compute_max_scale_factor()
{
    // compute a scaling factor
    double max_scale_factor = 1;
    for (int eid = 0; eid < g_num_edges; eid++)
    {
        Edge& e = g_edges[eid];
        double sf = e.total_flow / e.cap;
        max_scale_factor = max(max_scale_factor, sf);
    }
    cout << "--x edge scale factor " << max_scale_factor << endl;

    for (int cid = 0; cid < g_num_demands; cid++)
    {
        double commod_flow = 0;
        Commodity& c = g_commods[cid];
        double sf = c.flow_alloc / c.demand;
        max_scale_factor = max(max_scale_factor, sf);
    }
    cout << "--x max(edge,commod) scale factor " << max_scale_factor << endl;

    return max_scale_factor;
}

void check_flow_conservation()
{
    // assert
    for (int cid = 0; cid < g_num_demands; cid++)
    {
        double commod_flow = 0;
        Commodity& c = g_commods[cid];
        Node& source = g_nodes[c.n_source];
        map<int, int>::iterator miii;
        for (miii = source.out_nbrs.begin(); miii != source.out_nbrs.end(); miii++)
        {
            int eid = (*miii).second;
            commod_flow += g_edges[eid].flow[cid];
        }

        if (abs(c.flow_alloc - commod_flow) > 0.01 * c.flow_alloc)
            cout << "Warn!!! src flow not being conserved: " << cid << " " << c.flow_alloc << " " << commod_flow;

        Node& target = g_nodes[c.n_target];
        commod_flow = 0;
        for (miii = target.in_nbrs.begin(); miii != target.in_nbrs.end(); miii++)
        {
            int eid = (*miii).second;
            commod_flow += g_edges[eid].flow[cid];
        }

        if (abs(c.flow_alloc - commod_flow) > 0.01 * c.flow_alloc)
            cout << "Warn!!! target flow not being conserved: " << cid << " " << c.flow_alloc << " " << commod_flow;
    }
}

double scale_flows(double sf)
{
    bool debug = false;
    double total_flow = 0;
    for (int cid = 0; cid < g_num_demands; cid++)
    {
        Commodity& c = g_commods[cid];
        c.flow_alloc /= sf;

        if (debug) cout << "--x commod[" << cid << "] flow= " << c.flow_alloc << endl;
        total_flow += c.flow_alloc;
    }

    return total_flow;
}

double get_timediff(struct timeval begin, struct timeval now)
{
    return ((now.tv_sec - begin.tv_sec) + (now.tv_usec - begin.tv_usec) / 1000000.0);
}

// there are several discrepancies between the text and the algorithm written out
// in figure 2.2 of fleischer
void run_fleischer_2_2(bool restrict_paths = false)
{
    struct timeval tv_start, tv_end, tv_now;

    gettimeofday(&tv_start, (struct timezone*) NULL);

    bool debug = false;
    double debug_flow_alloc_since_last_print = 0;
    int debug_num_allocs_since_last_print = 0;

    double delta = (1 + g_epsilon) * pow((1 + g_epsilon) * (g_num_nodes + g_num_demands), -1.0 / g_epsilon);

    double* commod_weights = new double[g_num_demands];
    for (int c_id = 0; c_id < g_num_demands; c_id++)
        commod_weights[c_id] = delta;

    double* e_weights = new double[g_num_edges];
    for (int e_id = 0; e_id < g_num_edges; e_id++)
        e_weights[e_id] = delta;

    int iter_max = floor(log((1 + g_epsilon) / delta) / log(1 + g_epsilon));
    cout << "Fleischer setup: delta = " << delta <<
        " iter_max = " << iter_max <<
        endl;

    double path_length_cutoff = delta;
    vector<int> edge_ids_on_path;

    for (int r = 1; r <= iter_max; r++)
    {
        path_length_cutoff *= (1 + g_epsilon);

        if (debug || r % 100 == 0)
        {
            gettimeofday(&tv_now, (struct timezone*)NULL);
            cout << "iter " << r
                << " cutoff " << path_length_cutoff
                << " elapsed: " << get_timediff(tv_start, tv_now) << "s"
                << " #calls: d=" << g_num_dijkstra_calls << " s=" << g_num_shortest_path_calls
                << " since last print: FlowAlloc=" << debug_flow_alloc_since_last_print
                << " numAlloc=" << debug_num_allocs_since_last_print
                << endl;
            debug_flow_alloc_since_last_print = 0;
            debug_num_allocs_since_last_print = 0;
        }

        for (int commod_id = 0; commod_id < g_num_demands; commod_id++)
        {
            Commodity& c = g_commods[commod_id];

            bool can_allocate_more;
            do
            {
                can_allocate_more = false;

                double path_length = commod_weights[commod_id];
                double min_cap = c.demand;

                int s = c.n_source;
                int t = c.n_target;
                // if (r == 141) cout << s << " -> " << t << endl;
                if (restrict_paths)
                    run_simple_shortest_paths(s, e_weights, t, false);
                else 
                    run_dijkstra(s, e_weights, t);
                
                edge_ids_on_path.clear();

                if (restrict_paths)
                {
                    pair<int, double>& tuid = g_best_path[s][t];
                    int min_path_id = tuid.first;
                    if (min_path_id == -1) {
                      continue;
                    }
                    int* path = g_paths[s][t][min_path_id];
                    double x = 0;
                    for (int index = 1; index <= path[0]; index++)
                    {
                        int eid = path[index];
                        if (!(eid >= 0 && eid < g_num_edges))
                        {
                            cout << s << " -> " << t << endl;
                            cout << index << " " << path[0] << endl;
                            cout << eid << endl;
                        }
                        assertm(eid >= 0 && eid < g_num_edges, "not valid edge");

                        Edge& e = g_edges[eid];
                        edge_ids_on_path.push_back(e.id);
                        min_cap = min(min_cap, e.cap);
                        x += e_weights[e.id];
                    }
                    path_length += tuid.second;

                    if (abs(x - tuid.second) > 0.01 * tuid.second)
                    {
                        cout << "--------" << s << " -> " << t << endl;
                        cout << "mismatch in weights; x= " << x << " ts= " << tuid.second << endl;
                        cout << "edges_on_path= " << print_vector(edge_ids_on_path) << endl;
                    }
                }                
                else
                    for (int curr = t; curr != s; )
                    {
                        Node& c = g_nodes[curr];
                        Edge& e = g_edges[c.in_nbrs[c.pre_id]];

                        path_length += e_weights[e.id];
                        min_cap = min(min_cap, e.cap);

                        curr = c.pre_id;

                        edge_ids_on_path.push_back(e.id);
                    }

                can_allocate_more = path_length < min(1.0, path_length_cutoff);
                if (can_allocate_more)
                {
                    commod_weights[commod_id] *= (1 + (min_cap * g_epsilon / c.demand));
                    c.flow_alloc += min_cap;

                    // if (r == 141) cout << "path=" << print_vector(edge_ids_on_path) << endl;

                    for (vector<int>::iterator eiopi = edge_ids_on_path.begin(); 
                        eiopi != edge_ids_on_path.end(); 
                        eiopi++)
                    {
                        int e_id = *(eiopi);
                        Edge& e = g_edges[e_id];

                        /*
                        double factor = (1 + (min_cap * g_epsilon / e.cap));
                        double prev = e_weights[e_id];
                        if (prev * factor > 1)
                        {
                            cout << "WARN!!! prev= " << prev << " factor= " << factor << endl;
                        }
                        */

                        e_weights[e_id] *= (1 + (min_cap * g_epsilon / e.cap));
                        // assertm(e_weights[e_id] <= 1, "edge_weight too large");

                        e.flow[commod_id] += min_cap;
                        e.total_flow += min_cap;

                        debug_flow_alloc_since_last_print += min_cap;
                        debug_num_allocs_since_last_print++;
                    }
                }

            } while (can_allocate_more);
        }
    }

    gettimeofday(&tv_end, (struct timezone*) NULL);

    check_flow_conservation();
    double tf = scale_flows(compute_max_scale_factor());
    cout << "Total flow = " << tf << 
        " #calls = dijkstra: " << g_num_dijkstra_calls << 
        " ssp: " << g_num_shortest_path_calls <<
        " time elapsed: " << get_timediff(tv_start, tv_end) <<
        endl;
}


void run_fleischer()
{
    struct timeval tv_start, tv_end, tv_now;

    gettimeofday(&tv_start, (struct timezone*) NULL);

    bool debug = false;
    double debug_flow_alloc_since_last_print = 0;
    int debug_num_allocs_since_last_print = 0;

    double delta = (1 + g_epsilon) * pow((1 + g_epsilon) * (g_num_nodes+g_num_demands), -1.0/g_epsilon);

    double *commod_weights = new double[g_num_demands];
    for (int c_id = 0; c_id < g_num_demands; c_id++)
        commod_weights[c_id] = delta;

    double* e_weights = new double[g_num_edges];
    for (int e_id = 0; e_id < g_num_edges; e_id++)
        e_weights[e_id] = delta;

    int iter_max = floor(log((1 + g_epsilon) / delta) / log(1 + g_epsilon));
    cout << "Fleischer setup: delta = " << delta << 
        " iter_max = " << iter_max <<
        endl;

    double path_length_cutoff = delta;
    vector<int> edges_on_path;
    for (int r = 1; r <= iter_max; r++)
    {
        path_length_cutoff *= (1 + g_epsilon);

        if (debug || r % 100 == 0)
        {
            gettimeofday(&tv_now, (struct timezone*)NULL);
            cout << "iter " << r 
                << " cutoff " << path_length_cutoff 
                << " elapsed: " << get_timediff(tv_start, tv_now) << "s" 
                << " #calls: " << g_num_dijkstra_calls << " dijkstras"
                << " since last print: FlowAlloc=" << debug_flow_alloc_since_last_print 
                << " numAlloc=" << debug_num_allocs_since_last_print
                << endl;
            debug_flow_alloc_since_last_print = 0;
            debug_num_allocs_since_last_print = 0;
        }

        // for each source, consider commodities going to any target from this source
        for (int n_id = 0; n_id < g_num_nodes; n_id++)
        {            
            Node& source = g_nodes[n_id];
            bool this_source_had_an_increment;

            do
            {
                this_source_had_an_increment = false;

                // compute shortest paths from source to all targets (todo: restrict to those with active commods)
                run_dijkstra(n_id, e_weights);

                for (list<int>::iterator it = source.from_commods.begin(); 
                    it != source.from_commods.end(); 
                    ++it)
                {
                    int j = *it; // commod id
                    Commodity& commod = g_commods[j];

                    bool this_commod_can_increment;
                    do
                    {
                        this_commod_can_increment = false;

                        double path_length = commod_weights[j];
                        double min_cap = commod.demand;
                        edges_on_path.clear();
                        for (int curr_node = commod.n_target; curr_node != commod.n_source; )
                        {
                            Node& curr = g_nodes[curr_node];
                            int prev = g_nodes[curr_node].pre_id;
                            Edge& e = g_edges[curr.in_nbrs[prev]];

                            path_length += e_weights[e.id];
                            min_cap = min(min_cap, e.cap);

                            // move to prev
                            curr_node = prev;
                            edges_on_path.push_back(e.id);
                        }

                        if (debug) cout << "C" << j << "; path= " << print_vector(edges_on_path) << endl;

                        if (path_length < min(1.0, path_length_cutoff))
                        {
                            this_commod_can_increment = true;
                            this_source_had_an_increment = true;

                            // update the weights and assign flow
                            commod_weights[j] *= (1 + (min_cap * g_epsilon / commod.demand));
                            commod.flow_alloc += min_cap;
                            
                            debug_num_allocs_since_last_print++;
                            debug_flow_alloc_since_last_print += min_cap;
                            
                            for(vector<int>::iterator vepi = edges_on_path.begin();
                                vepi != edges_on_path.end(); 
                                vepi++)
                            {
                                Edge& e = g_edges[*vepi];

                                e_weights[e.id] *= (1 + (min_cap * g_epsilon / e.cap));

                                e.flow[j] += min_cap;
                                e.total_flow += min_cap;
                            }
                        }
                    } while (this_commod_can_increment); // push more along the path chosen for this commod
                } // push more along the dijkstra path for some other commod (note edge weights may have changed)
            } while (this_source_had_an_increment); // we give up when no commod can increment
        }
    }

    gettimeofday(&tv_end, (struct timezone*) NULL);

    check_flow_conservation();

    double total_flow = scale_flows(compute_max_scale_factor());
    double runtime_s = ((tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0);
    cout << "Total flow = " << total_flow 
        << " Runtime= " << runtime_s << " s" 
        << " #Dijkstras= " << g_num_dijkstra_calls
        << " since last print: FlowAlloc=" << debug_flow_alloc_since_last_print
        << " numAlloc=" << debug_num_allocs_since_last_print
        << endl;
}

int main(int argc, char** argv)
{
    bool debug = true;

    if (argc > 4 || argc < 2)
    {
        cout << "Usage: [] [switch] <inputfile> [epsilon] " << endl;
        cout << "switch choices are " << endl;
        cout << "    -f // run fleischer algo interpreted from paper's text" << endl;
        cout << "    -22 // run pseudocode 2.2 of fleischer paper" << endl;
        cout << "    -p // restrict paths; paths should be present at the bottom of input file" << endl;
        cout << " valid switches are -f -22 -22p" << endl;
        return (1);
    }

    // parse arguments
    bool runMyFleischer = false, runFleischer2_2 = false, restrictPaths = false;
    int nextarg = 1;
    if (string(argv[nextarg]).rfind("-") == 0)
    {
        string _switch = string(argv[nextarg]);

        if (debug) cout << "switch= " << _switch << endl;

        runMyFleischer = _switch.rfind("f") != string::npos;
        runFleischer2_2 = _switch.rfind("22") != string::npos;
        restrictPaths = _switch.rfind("p") != string::npos;
        nextarg++;

    }
    else
    {
        runMyFleischer = true;
    }

    char* fname = argv[nextarg];
    if (argc >= nextarg + 2)
        stringstream(argv[nextarg + 1]) >> g_epsilon;
    else
        g_epsilon = 0.1;

    // cout << "reading from " << fname << endl;
    // exit(0);

    // read inputs
    try
    {
        ReadTopoAndDemands(fname, restrictPaths);
    }
    catch (int e)
    {
        cout << "Exception reading from " << fname << " excep= " << e << endl;
        return (2);
    }

    cout
        << "#nodes " << g_num_nodes
        << " #edges " << g_num_edges
        << " #demands " << g_num_demands
        << " #paths " << g_num_paths
        << endl;

    if (runMyFleischer) run_fleischer();
    else if (runFleischer2_2) run_fleischer_2_2(restrictPaths);
    else cout << "nothing to do?" << endl;
}
