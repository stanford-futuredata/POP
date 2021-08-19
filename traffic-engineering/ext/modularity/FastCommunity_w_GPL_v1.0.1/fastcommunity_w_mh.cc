////////////////////////////////////////////////////////////////////////
// --- COPYRIGHT NOTICE ---------------------------------------------
// FastCommunityMH - infers community structure of networks
// Copyright (C) 2004 Aaron Clauset
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// See http://www.gnu.org/licenses/gpl.txt for more details.
// 
////////////////////////////////////////////////////////////////////////
// Author       : Aaron Clauset  (aaron@cs.unm.edu)				//
// Location     : U. Michigan, U. New Mexico						//
// Time         : January-August 2004							//
// Collaborators: Dr. Cris Moore (moore@cs.unm.edu)				//
//              : Dr. Mark Newman (mejn@umich.edu)				//
////////////////////////////////////////////////////////////////////////
// --- DEEPER DESCRIPTION ---------------------------------------------
//  see http://www.arxiv.org/abs/cond-mat/0408187 for more information
// 
//  - read network structure from data file (see below for constraints)
//  - builds dQ, H and a data structures
//  - runs new fast community structure inference algorithm
//  - records Q(t) function to file
//  - (optional) records community structure (at t==cutstep)
//  - (optional) records the list of members in each community (at t==cutstep)
//
////////////////////////////////////////////////////////////////////////
// --- PROGRAM USAGE NOTES ---------------------------------------------
// This program is rather complicated and requires a specific kind of input,
// so some notes on how to use it are in order. Mainly, the program requires
// a specific structure input file (.pairs) that has the following characteristics:
//  
//  1. .pairs is a list of tab-delimited pairs of numeric indices, e.g.,
//		"54\t91\n"
//  2. the network described is a SINGLE COMPONENT
//  3. there are NO SELF-LOOPS or MULTI-EDGES in the file; you can use
//     the 'netstats' utility to extract the giantcomponent (-gcomp.pairs)
//     and then use that file as input to this program
//  4. the MINIMUM NODE ID = 0 in the input file; the maximum can be
//     anything (the program will infer it from the input file)
// 
// Description of commandline arguments
// -f <filename>    give the target .pairs file to be processed
// -l <text>		the text label for this run; used to build output filenames
// -t <int>		timer period for reporting progress of file input to screen
// -s			calculate and record the support of the dQ matrix
// -v --v ---v		differing levels of screen output verbosity
// -o <directory>   directory for file output
// -c <int>		record the aglomerated network at step <int>
// 
////////////////////////////////////////////////////////////////////////
// Change Log:
// 2006-02-06: 1) modified readInputFile to be more descriptive of its actions
//             2) removed -o functionality; program writes output to directory
//             of input file. (also removed -h option of command line)
// 2006-10-13: 3) Janne Aukia (jaukia@cc.hut.fi) suggested changes to the 
//             mergeCommunities() function here (see comments in that function),
//             and an indexing adjustment in printHeapTop10() in maxheap.h.
//
////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#include "maxheap.h"
#include "vektor.h"

using namespace std;

// ------------------------------------------------------------------------------------
// Edge object - defined by a pair of vertex indices and *edge pointer to next in linked-list
class edge {
public:
	int     so;					// originating node
	int     si;					// terminating node
	int     we;					// edge weight
	edge    *next;					// pointer for linked list of edges
	
	edge();						// default constructor
	~edge();						// default destructor
};
edge::edge()  { so = 0; si = 0; next = NULL; }
edge::~edge() {}

// ------------------------------------------------------------------------------------
// Nodenub object - defined by a *node pointer and *node pointer 
struct nodenub {
	sktuple	*heap_ptr;			// pointer to node(max,i,j) in max-heap of row maxes
	vektor    *v;					// pointer stored vector for (i,j)
};

// ------------------------------------------------------------------------------------
// sktuple object - defined by an real value and (row,col) indices
#if !defined(TUPLE_INCLUDED)
#define TUPLE_INCLUDED
struct sktuple {
	double    m;					// stored value
	int		i;					// row index
	int		j;					// column index
	int		k;					// heap index
};
#endif

// ordered pair structures (handy in the program)
struct apair { int x; int y; };
#if !defined(DPAIR_INCLUDED)
#define DPAIR_INCLUDED
class dpair {
public:
	int x; double y; dpair *next;
	dpair(); ~dpair();
};
dpair::dpair()  { x = 0; y = 0.0; next = NULL; }
dpair::~dpair() {}
#endif

// ------------------------------------------------------------------------------------
// List object - simple linked list of integers
class list {
public:
	int		index;				// node index
	list		*next;				// pointer to next element in linked list
	list();   ~list();
};
list::list()  { index= 0; next = NULL; }
list::~list() {}

// ------------------------------------------------------------------------------------
// Community stub object - stub for a community list
class stub {
public:
	bool		valid;				// is this community valid?
	int		size;				// size of community
	list		*members;				// pointer to list of community members
	list		*last;				// pointer to end of list
	stub();   ~stub();
};
stub::stub()  { valid = false; size = 0; members = NULL; last = NULL; }
stub::~stub() {
	list *current;
	if (members != NULL) {
		current = members;
		while (current != NULL) { members = current->next; delete current; current = members; }
	}
}

// ------------------------------------------------------------------------------------
// FUNCTION DECLARATIONS --------------------------------------------------------------

void buildDeltaQMatrix();
void buildFilenames();
void dqSupport();
void groupListsSetup();
void groupListsStats();
void groupListsUpdate(const int x, const int y);
void mergeCommunities(int i, int j);
bool parseCommandLine(int argc,char * argv[]);
void readInputFile();
void recordGroupLists();
void recordNetwork();

// ------------------------------------------------------------------------------------
// PROGRAM PARAMETERS -----------------------------------------------------------------

struct netparameters {
	int			n;				// number of nodes in network
	int			m;				// number of edges in network
	int			w;				// total weight of edges in network
	int			maxid;			// maximum node id
	int			minid;			// minimum node id
}; netparameters    gparm;

struct groupstats {
	int			numgroups;		// number of groups
	double		meansize;			// mean size of groups
	int			maxsize;			// size of largest group
	int			minsize;			// size of smallest group
	double		*sizehist;		// distribution of sizes
}; groupstats		gstats;

struct outparameters {
	short int		textFlag;			// 0: no console output
								// 1: writes file outputs
	bool			suppFlag;			// T: no support(t) file
								// F: yes support(t) file
	short int		fileFlag;			// 
	string		filename;			// name of input file
	string		d_in;			// (dir ) directory for input file
	string		d_out;			// (dir ) director for output file
	string		f_parm;			// (file) parameters output
	string		f_input;			// (file) input data file
	string		f_joins;			// (file) community hierarchy
	string		f_support;		// (file) dQ support as a function of time
	string		f_net;			// (file) .wpairs file for .cutstep network
	string		f_group;			// (file) .list of indices in communities at .cutstep
	string		f_gstats;			// (file) distribution of community sizes at .cutstep
	string		s_label;			// (temp) text label for run
	string		s_scratch;		// (temp) text for building filenames
	int			timer;			// timer for displaying progress reports 
	bool			timerFlag;		// flag for setting timer
	int			cutstep;			// step at which to record aglomerated network
}; outparameters	ioparm;

// ------------------------------------------------------------------------------------
// ----------------------------------- GLOBAL VARIABLES -------------------------------

char		pauseme;
edge		*e;				// initial adjacency matrix (sparse)
edge		*elist;			// list of edges for building adjacency matrix
nodenub   *dq;				// dQ matrix
maxheap   *h;				// heap of values from max_i{dQ_ij}
double    *Q;				// Q(t)
dpair     Qmax;			// maximum Q value and the corresponding time t
double    *a;				// A_i
apair	*joins;			// list of joins
stub		*c;				// link-lists for communities

enum {NONE};

int    supportTot;
double supportAve;

// ------------------------------------------------------------------------------------
// ----------------------------------- MAIN PROGRAM -----------------------------------
int main(int argc,char * argv[]) {

	// default values for parameters which may be modified from the commandline
	ioparm.timer     = 20;
	ioparm.fileFlag  = NONE;
	ioparm.suppFlag  = false;
	ioparm.textFlag  = 0;
	ioparm.filename  = "community.pairs";
	ioparm.s_label   = "a";
	time_t t1;	t1 = time(&t1);
	time_t t2;	t2 = time(&t2);

        clock_t start_time = clock();
	
	// ----------------------------------------------------------------------
	// Parse the command line, build filenames and then import the .pairs file
	cout << "\nFast Community Inference.\n";
	cout << "Copyright (c) 2004 by Aaron Clauset (aaron@cs.unm.edu)\n";
	if (parseCommandLine(argc, argv)) {} else { return 0; }
	cout << "\nimporting: " << ioparm.filename << endl;    // note the input filename
	buildFilenames();								// builds filename strings
	readInputFile();								// gets adjacency matrix data
	
	// ----------------------------------------------------------------------
	// Allocate data structures for main loop
	a     = new double [gparm.maxid];
	Q     = new double [gparm.n+1];
	joins = new apair  [gparm.n+1];
	for (int i=0; i<gparm.maxid; i++) { a[i] = 0.0; }
	for (int i=0; i<gparm.n+1;   i++) { Q[i] = 0.0; joins[i].x = 0; joins[i].y = 0; }
	int t = 1;
	Qmax.y = -4294967296.0;  Qmax.x = 0;
	if (ioparm.cutstep > 0) { groupListsSetup(); }		// will need to track agglomerations
	
	cout << "now building initial dQ[]" << endl;
	buildDeltaQMatrix();							// builds dQ[] and h
	
	// initialize f_joins, f_support files
	ofstream fjoins(ioparm.f_joins.c_str(), ios::trunc);
	fjoins << -1 << "\t" << -1 << "\t" << Q[0] << "\t0\n";
	fjoins.close();

	if (ioparm.suppFlag) {
		ofstream fsupp(ioparm.f_support.c_str(), ios::trunc);
		dqSupport();
		fsupp << 0 << "\t" << supportTot << "\t" << supportAve << "\t" << 0 << "\t->\t" << 0 << "\n";
		fsupp.close();
	}
	
	// ----------------------------------------------------------------------
	// Start FastCommunity algorithm
	cout << "starting algorithm now." << endl;
	sktuple  dQmax, dQnew;
	int isupport, jsupport;
	while (h->heapSize() > 2) {
		
		// ---------------------------------
		// Find largest dQ
		if (ioparm.textFlag > 0) { h->printHeapTop10(); cout << endl; }
		dQmax = h->popMaximum();					// select maximum dQ_ij // convention: insert i into j
		if (dQmax.m < -4000000000.0) { break; }		// no more joins possible
		cout << "Q["<<t-1<<"] = "<<Q[t-1];
		
		// ---------------------------------
		// Merge the chosen communities
		cout << "\tdQ = " << dQmax.m << "\t  |H| = " << h->heapSize() << "\n";
		if (dq[dQmax.i].v == NULL || dq[dQmax.j].v == NULL) {
			cout << "WARNING: invalid join (" << dQmax.i << " " << dQmax.j << ") found at top of heap\n"; cin >> pauseme;
		}
		isupport = dq[dQmax.i].v->returnNodecount();
		jsupport = dq[dQmax.j].v->returnNodecount();
		if (isupport < jsupport) {
			cout << "  join: " << dQmax.i << " -> " << dQmax.j << "\t";
			cout << "(" << isupport << " -> " << jsupport << ")\n";
			mergeCommunities(dQmax.i, dQmax.j);	// merge community i into community j
			joins[t].x = dQmax.i;				// record merge of i(x) into j(y)
			joins[t].y = dQmax.j;				// 
		} else {								// 
			cout << "  join: " << dQmax.i << " <- " << dQmax.j << "\t";
			cout << "(" << isupport << " <- " << jsupport << ")\n";
			dq[dQmax.i].heap_ptr = dq[dQmax.j].heap_ptr; // take community j's heap pointer
			dq[dQmax.i].heap_ptr->i = dQmax.i;			//   mark it as i's
			dq[dQmax.i].heap_ptr->j = dQmax.j;			//   mark it as i's
			mergeCommunities(dQmax.j, dQmax.i);	// merge community j into community i
			joins[t].x = dQmax.j;				// record merge of j(x) into i(y)
			joins[t].y = dQmax.i;				// 
		}									// 
		Q[t] = dQmax.m + Q[t-1];					// record Q(t)
		
		// ---------------------------------
		// Record join to file
		ofstream fjoins(ioparm.f_joins.c_str(), ios::app);   // open file for writing the next join
		fjoins << joins[t].x-1 << "\t" << joins[t].y-1 << "\t";	// convert to external format
		if ((Q[t] > 0.0 && Q[t] < 0.0000000000001) || (Q[t] < 0.0 && Q[t] > -0.0000000000001))
			{ fjoins << 0.0; } else { fjoins << Q[t]; }
		fjoins << "\t" << t << "\n";
		fjoins.close();
		// Note that it is the .joins file which contains both the dendrogram and the corresponding
		// Q values. The file format is tab-delimited columns of data, where the columns are:
		// 1. the community which grows
		// 2. the community which was absorbed
		// 3. the modularity value Q after the join
		// 4. the time step value
		
		// ---------------------------------
		// If cutstep valid, then do some work
		if (t <= ioparm.cutstep) { groupListsUpdate(joins[t].x, joins[t].y); }
		if (t == ioparm.cutstep) { recordNetwork(); recordGroupLists(); groupListsStats(); }

		// ---------------------------------
		// Record the support data to file
		if (ioparm.suppFlag) {
			dqSupport();
			ofstream fsupp(ioparm.f_support.c_str(), ios::app);
			// time   remaining support   mean support   support_i --   support_j
			fsupp << t << "\t" << supportTot << "\t" << supportAve << "\t" << isupport;
			if (isupport < jsupport) { fsupp  << "\t->\t"; }
			else { fsupp << "\t<-\t"; }
			fsupp << jsupport << "\n";
			fsupp.close();
		}
		if (Q[t] > Qmax.y) { Qmax.y = Q[t]; Qmax.x = t; }
		
		t++;									// increment time
	} // ------------- end community merging loop
	cout << "Q["<<t-1<<"] = "<<Q[t-1] << endl;
	
	// ----------------------------------------------------------------------
	// Record some results
	t1 = time(&t1);
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---MODULARITY---\n";
	fout << "MAXQ------:\t" << Qmax.y  << "\n";
	fout << "STEP------:\t" << Qmax.x  << "\n";
	fout << "EXIT------:\t" << asctime(localtime(&t1));
	fout.close();

        cout << "Total Time: " <<  ((double) (clock() - start_time)) / CLOCKS_PER_SEC << " seconds" << endl;

	cout << "exited safely" << endl;
	return 1;
}


// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
// FUNCTION DEFINITIONS --------------------------------------------------------------- //

void buildDeltaQMatrix() {
	
	// Given that we've now populated a sparse (unordered) adjacency matrix e (e), 
	// we now need to construct the intial dQ matrix according to the definition of dQ
	// which may be derived from the definition of modularity Q:
	//    Q(t) = \sum_{i} (e_{ii} - a_{i}^2) = Tr(e) - ||e^2||
	// thus dQ is
	//    dQ_{i,j} = 2* ( e_{i,j} - a_{i}a_{j} )
	//    where a_{i} = \sum_{j} e_{i,j} (i.e., the sum over the ith row)
	// To create dQ, we must insert each value of dQ_{i,j} into a binary search tree,
	// for the jth column. That is, dQ is simply an array of such binary search trees,
	// each of which represents the dQ_{x,j} adjacency vector. Having created dQ as
	// such, we may happily delete the matrix e in order to free up more memory.
	// The next step is to create a max-heap data structure, which contains the entries
	// of the following form (value, s, t), where the heap-key is 'value'. Accessing the
	// root of the heap gives us the next dQ value, and the indices (s,t) of the vectors
	// in dQ which need to be updated as a result of the merge.
	
	// First we compute e_{i,j}, and the compute+store the a_{i} values. These will be used
	// shortly when we compute each dQ_{i,j}.
	edge   *current;
	int    deg[gparm.maxid];
	double  eij = (double)(0.5/gparm.m);				// intially each e_{i,j} = 1/m
	for (int i=1; i<gparm.maxid; i++) {				// for each row
		a[i]   = 0.0;								// 
		if (e[i].so != 0) {							//    ensure it exists
			current = &e[i];						//    grab first edge
			deg[i]  = 0;							// 
			while (current != NULL) {
				a[i] += (current->we)/(2.0*gparm.w);	//    update a[i]
				deg[i]++;							//	 increment degree count
				current = current->next;				//
			}
			Q[0] += -1.0*a[i]*a[i];					// calculate initial value of Q
		} else { deg[i] = -1; }						// 
	}

	// now we create an empty (ordered) sparse matrix dq[]
	dq = new nodenub [gparm.maxid];						// initialize dq matrix
	for (int i=0; i<gparm.maxid; i++) {					// 
		dq[i].heap_ptr = NULL;							// no pointer in the heap at first
//		if (e[i].so != 0) { dq[i].v = new vektor(2+(int)floor(gparm.m*a[i])); }
		if (e[i].so != 0) { dq[i].v = new vektor(2+deg[i]); }
		else {			dq[i].v = NULL; }
	}
	h = new maxheap(gparm.n);							// allocate max-heap of size = number of nodes
	
	// Now we do all the work, which happens as we compute and insert each dQ_{i,j} into 
	// the corresponding (ordered) sparse vector dq[i]. While computing each dQ for a
	// row i, we track the maximum dQmax of the row and its (row,col) indices (i,j). Upon
	// finishing all dQ's for a row, we insert the sktuple into the max-heap hQmax. That
	// insertion returns the itemaddress, which we then store in the nodenub heap_ptr for 
	// that row's vector.
	double    dQ;
	sktuple	dQmax;										// for heaping the row maxes
	sktuple*    itemaddress;									// stores address of item in maxheap

	for (int i=1; i<gparm.maxid; i++) {
		if (e[i].so != 0) {
			current = &e[i];								// grab first edge
			eij     = (current->we)/(2.0*gparm.w);				// compute this eij
			dQ      = 2.0*(eij-(a[current->so]*a[current->si]));   // compute its dQ
			dQmax.m = dQ;									// assume it is maximum so far
			dQmax.i = current->so;							// store its (row,col)
			dQmax.j = current->si;							// 
			dq[i].v->insertItem(current->si, dQ);				// insert its dQ
			while (current->next != NULL) {					// 
				current = current->next;						// step to next edge
				eij     = (current->we)/(2.0*gparm.w);			// compute this eij
				dQ = 2.0*(eij-(a[current->so]*a[current->si]));	// compute new dQ
				if (dQ > dQmax.m) {							// if dQ larger than current max
					dQmax.m = dQ;							//    replace it as maximum so far
					dQmax.j = current->si;					//    and store its (col)
				}
				dq[i].v->insertItem(current->si, dQ);			// insert it into vector[i]
			}
			dq[i].heap_ptr = h->insertItem(dQmax);				// store the pointer to its loc in heap
		}
	}

	delete [] elist;								// free-up adjacency matrix memory in two shots
	delete [] e;									// 
	return;
}

// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------ //

void buildFilenames() {

	ioparm.f_input   = ioparm.d_in  + ioparm.filename;
	ioparm.f_parm    = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".info";
	ioparm.f_joins   = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".joins";
	ioparm.f_support = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".supp";
	ioparm.f_net     = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".wpairs";
	ioparm.f_group   = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".groups";
	ioparm.f_gstats  = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".hist";
	
	if (true) { ofstream flog(ioparm.f_parm.c_str(), ios::trunc); flog.close(); }
	time_t t; t = time(&t);
	ofstream flog(ioparm.f_parm.c_str(), ios::app);
	flog << "FASTCOMMUNITY_INFERENCE_ALGORITHM\n";
	flog << "START-----:\t" << asctime(localtime(&t));
	flog << "---FILES--------\n";
	flog << "DIRECTORY-:\t" << ioparm.d_out		<< "\n";
	flog << "F_IN------:\t" << ioparm.filename   << "\n";
	flog << "F_JOINS---:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".joins" << "\n";
	flog << "F_INFO----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".info"  << "\n";
	if (ioparm.suppFlag) {
		flog << "F_SUPP----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".supp" << "\n"; }
	if (ioparm.cutstep>0) {
		flog << "F_NET-----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".wpairs" << "\n";
		flog << "F_GROUPS--:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".groups" << "\n";
		flog << "F_GDIST---:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".hist"   << "\n";
	}
	flog.close();
	
	return;
}

// ------------------------------------------------------------------------------------
// returns the support of the dQ[]

void dqSupport() {
	int    total = 0;
	int    count = 0;
	for (int i=0; i<gparm.maxid; i++) {
		if (dq[i].heap_ptr != NULL) { total += dq[i].v->returnNodecount(); count++; }
	}
	supportTot = total;
	supportAve = total/(double)count;
	return;
}

// ------------------------------------------------------------------------------------

void groupListsSetup() {
	
	list *newList;
	c = new stub [gparm.maxid];
	for (int i=0; i<gparm.maxid; i++) {
		if (e[i].so != 0) {								// note: internal indexing
			newList = new list;							// create new community member
			newList->index = i;							//    with index i
			c[i].members   = newList;					// point ith community at newList
			c[i].size		= 1;							// point ith community at newList
			c[i].last		= newList;					// point last[] at that element too
			c[i].valid	= true;						// mark as valid community
		}
	}
	
	return;
}

// ------------------------------------------------------------------------------------
// function for computing statistics on the list of groups

void groupListsStats() {

	gstats.numgroups = 0;
	gstats.maxsize   = 0;
	gstats.minsize   = gparm.maxid;
	double count     = 0.0;
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			gstats.numgroups++;							// count number of communities
			count += 1.0;
			if (c[i].size > gstats.maxsize) { gstats.maxsize = c[i].size; }  // find biggest community
			if (c[i].size < gstats.minsize) { gstats.minsize = c[i].size; }  // find smallest community
			// compute mean group size
			gstats.meansize = (double)(c[i].size)/count + (((double)(count-1.0)/count)*gstats.meansize);
		}
	}
	
	count = 0.0;
	gstats.sizehist = new double [gstats.maxsize+1];
	for (int i=0; i<gstats.maxsize+1; i++) { gstats.sizehist[i] = 0; }
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			gstats.sizehist[c[i].size] += 1.0;					// tabulate histogram of sizes
			count += 1.0;
		}
	}
	// convert histogram to pdf, and write it to disk
	for (int i=0; i<gstats.maxsize+1; i++) { gstats.sizehist[i] = gstats.sizehist[i]/count; }
	ofstream fgstat(ioparm.f_gstats.c_str(), ios::trunc);
	for (int i=gstats.minsize; i<gstats.maxsize+1; i++) {
		fgstat << i << "\t" << gstats.sizehist[i] << "\n";
	}
	fgstat.close();
	
	// record some statistics
	time_t t1; t1 = time(&t1);
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---GROUPS-------\n";
	fout << "NUMGROUPS-:\t" << gstats.numgroups  << "\n";
	fout << "MINSIZE---:\t" << gstats.minsize    << "\n";
	fout << "MEANSIZE--:\t" << gstats.meansize   << "\n";
	fout << "MAXSIZE---:\t" << gstats.maxsize    << "\n";
	fout.close();
	
	return;
}

// ------------------------------------------------------------------------------------

void groupListsUpdate(const int x, const int y) {
	
	c[y].last->next = c[x].members;				// attach c[y] to end of c[x]
	c[y].last		 = c[x].last;					// update last[] for community y
	c[y].size		 += c[x].size;					// add size of x to size of y
	
	c[x].members   = NULL;						// delete community[x]
	c[x].valid	= false;						// 
	c[x].size		= 0;							// 
	c[x].last		= NULL;						// delete last[] for community x
	
	return;
}

// ------------------------------------------------------------------------------------

void mergeCommunities(int i, int j) {
	
	// To do the join operation for a pair of communities (i,j), we must update the dQ
	// values which correspond to any neighbor of either i or j to reflect the change.
	// In doing this, there are three update rules (cases) to follow:
	//  1. jix-triangle, in which the community x is a neighbor of both i and j
	//  2. jix-chain, in which community x is a neighbor of i but not j
	//  3. ijx-chain, in which community x is a neighbor of j but not i
	//
	// For the first two cases, we may make these updates by simply getting a list of
	// the elements (x,dQ) of [i] and stepping through them. If x==j, then we can ignore
	// that value since it corresponds to an edge which is being absorbed by the joined
	// community (i,j). If [j] also contains an element (x,dQ), then we have a triangle;
	// if it does not, then we have a jix-chain.
	//
	// The last case requires that we step through the elements (x,dQ) of [j] and update each
	// if [i] does not have an element (x,dQ), since that implies a ijx-chain.
	// 
	// Let d([i]) be the degree of the vector [i], and let k = d([i]) + d([j]). The running
	// time of this operation is O(k log k)
	//
	// Essentially, we do most of the following operations for each element of
	// dq[i]_x where x \not= j
	//  1.  add dq[i]_x to dq[j]_x (2 cases)
	//  2.  remove dq[x]_i
	//  3.  update maxheap[x]
	//  4.  add dq[i]_x to dq[x]_j (2 cases)
	//  5.  remove dq[j]_i
	//  6.  update maxheap[j]
	//  7.  update a[j] and a[i]
	//  8.  delete dq[i]
	
	dpair *list, *current, *temp;
	sktuple newMax;
	int t = 1;
	
	// -- Working with the community being inserted (dq[i])
	// The first thing we must do is get a list of the elements (x,dQ) in dq[i]. With this 
	// list, we can then insert each into dq[j].

	//	dq[i].v->printTree();
	list    = dq[i].v->returnTreeAsList();			// get a list of items in dq[i].v
	current = list;							// store ptr to head of list
	
	if (ioparm.textFlag>1) {
		cout << "stepping through the "<<dq[i].v->returnNodecount() << " elements of community " << i << endl;
	}
		
	// ---------------------------------------------------------------------------------
	// SEARCHING FOR JIX-TRIANGLES AND JIX-CHAINS --------------------------------------
	// Now that we have a list of the elements of [i], we can step through them to check if
	// they correspond to an jix-triangle, a jix-chain, or the edge (i,j), and do the appropriate
	// operation depending.
	
	while (current!=NULL) {						// insert list elements appropriately
		
		if (ioparm.textFlag>1) { cout << endl << "element["<<t<<"] from dq["<<i<<"] is ("<<current->x<<" "<<current->y<<")" << endl; }

		// If the element (x,dQ) is actually (j,dQ), then we can ignore it, since it will 
		// correspond to an edge internal to the joined community (i,j) after the join.
		if (current->x != j) {

			// Now we must decide if we have a jix-triangle or a jix-chain by asking if
			// [j] contains some element (x,dQ). If the following conditional is TRUE,
			// then we have a jix-triangle, ELSE it is a jix-chain.
			
			if (dq[j].v->findItem(current->x)) {
				// CASE OF JIX-TRIANGLE
				if (ioparm.textFlag>1) {
					cout << "  (0) case of triangle: e_{"<<current->x<<" "<<j<<"} exists" << endl;
					cout << "  (1) adding ("<<current->x<<" "<<current->y<<") to dq["<<current->x<<"] as ("<<j<<" "<<current->y<<")"<<endl;
				}
				
				// We first add (x,dQ) from [i] to [x] as (j,dQ), since [x] essentially now has
				// two connections to the joined community [j].
				if (ioparm.textFlag>1) {
					cout << "  (1) heapsize = " << dq[current->x].v->returnHeaplimit() << endl; 
					cout << "  (1) araysize = " << dq[current->x].v->returnArraysize() << endl; 
					cout << "  (1) vectsize = " << dq[current->x].v->returnNodecount() << endl; }
				dq[current->x].v->insertItem(j,current->y);			// (step 1)

				// Then we need to delete the element (i,dQ) in [x], since [i] is now a
				// part of [j] and [x] must reflect this connectivity.
				if (ioparm.textFlag>1) { cout << "  (2) now we delete items associated with "<< i << " in dq["<<current->x<<"]" << endl; }
				dq[current->x].v->deleteItem(i);					// (step 2)

				// After deleting an item, the tree may now have a new maximum element in [x],
				// so we need to check it against the old maximum element. If it's new, then
				// we need to update that value in the heap and reheapify.
				newMax = dq[current->x].v->returnMaxStored();		// (step 3)
				if (ioparm.textFlag>1) { cout << "  (3) and dq["<<current->x<<"]'s new maximum is (" << newMax.m <<" "<<newMax.j<< ") while the old maximum was (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				if (newMax.m > dq[current->x].heap_ptr->m || dq[current->x].heap_ptr->j==i) {
					h->updateItem(dq[current->x].heap_ptr, newMax);
					if (ioparm.textFlag>1) { cout << "  updated dq["<<current->x<<"].heap_ptr to be (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				}
// Change suggested by Janne Aukia (jaukia@cc.hut.fi) on 12 Oct 2006
				
				// Finally, we must insert (x,dQ) into [j] to note that [j] essentially now
				// has two connections with its neighbor [x].
				if (ioparm.textFlag>1) { cout << "  (4) adding ("<<current->x<<" "<<current->y<<") to dq["<<j<<"] as ("<<current->x<<" "<<current->y<<")"<<endl; }
				dq[j].v->insertItem(current->x,current->y);		// (step 4)
				
			} else {
				// CASE OF JIX-CHAIN
				
				// The first thing we need to do is calculate the adjustment factor (+) for updating elements.
				double axaj = -2.0*a[current->x]*a[j];
				if (ioparm.textFlag>1) {
					cout << "  (0) case of jix chain: e_{"<<current->x<<" "<<j<<"} absent" << endl;
					cout << "  (1) adding ("<<current->x<<" "<<current->y<<") to dq["<<current->x<<"] as ("<<j<<" "<<current->y+axaj<<")"<<endl;
				}
				
				// Then we insert a new element (j,dQ+) of [x] to represent that [x] has
				// acquired a connection to [j], which was [x]'d old connection to [i]
				dq[current->x].v->insertItem(j,current->y + axaj);	// (step 1)
				
				// Now the deletion of the connection from [x] to [i], since [i] is now
				// a part of [j]
				if (ioparm.textFlag>1) { cout << "  (2) now we delete items associated with "<< i << " in dq["<<current->x<<"]" << endl; }
				dq[current->x].v->deleteItem(i);					// (step 2)
				
				// Deleting that element may have changed the maximum element for [x], so we
				// need to check if the maximum of [x] is new (checking it against the value
				// in the heap) and then update the maximum in the heap if necessary.
				newMax = dq[current->x].v->returnMaxStored();		// (step 3)
				if (ioparm.textFlag>1) { cout << "  (3) and dq["<<current->x<<"]'s new maximum is (" << newMax.m <<" "<<newMax.j<< ") while the old maximum was (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				if (newMax.m > dq[current->x].heap_ptr->m || dq[current->x].heap_ptr->j==i) {
					h->updateItem(dq[current->x].heap_ptr, newMax);
					if (ioparm.textFlag>1) { cout << "  updated dq["<<current->x<<"].heap_ptr to be (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				}
// Change suggested by Janne Aukia (jaukia@cc.hut.fi) on 12 Oct 2006
					
				// Finally, we insert a new element (x,dQ+) of [j] to represent [j]'s new
				// connection to [x]
				if (ioparm.textFlag>1) { cout << "  (4) adding ("<<current->x<<" "<<current->y<<") to dq["<<j<<"] as ("<<current->x<<" "<<current->y+axaj<<")"<<endl; }
				dq[j].v->insertItem(current->x,current->y + axaj);	// (step 4)

			}    // if (dq[j].v->findItem(current->x))
			
		}    // if (current->x != j)
		
		temp    = current;
		current = current->next;						// move to next element
		delete temp;
		temp = NULL;
		t++;
	}    // while (current!=NULL)

	// We've now finished going through all of [i]'s connections, so we need to delete the element
	// of [j] which represented the connection to [i]
	if (ioparm.textFlag>1) {
		cout << endl;
		cout << "  whoops. no more elements for community "<< i << endl;
		cout << "  (5) now deleting items associated with "<<i<< " (the deleted community) in dq["<<j<<"]" << endl;
	}

	if (ioparm.textFlag>1) { dq[j].v->printTree(); }
	dq[j].v->deleteItem(i);						// (step 5)
	
	// We can be fairly certain that the maximum element of [j] was also the maximum
	// element of [i], so we need to check to update the maximum value of [j] that
	// is in the heap.
	newMax = dq[j].v->returnMaxStored();			// (step 6)
	if (ioparm.textFlag>1) { cout << "  (6) dq["<<j<<"]'s old maximum was (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
	h->updateItem(dq[j].heap_ptr, newMax);
	if (ioparm.textFlag>1) { cout << "      dq["<<j<<"]'s new maximum  is (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
	if (ioparm.textFlag>1) { dq[j].v->printTree(); }
	
	// ---------------------------------------------------------------------------------
	// SEARCHING FOR IJX-CHAINS --------------------------------------------------------
	// So far, we've treated all of [i]'s previous connections, and updated the elements
	// of dQ[] which corresponded to neighbors of [i] (which may also have been neighbors
	// of [j]. Now we need to update the neighbors of [j] (as necessary)
	
	// Again, the first thing we do is get a list of the elements of [j], so that we may
	// step through them and determine if that element constitutes an ijx-chain which
	// would require some action on our part.
	list = dq[j].v->returnTreeAsList();			// get a list of items in dq[j].v
	current = list;							// store ptr to head of list
	t       = 1;
	if (ioparm.textFlag>1) { cout << "\nstepping through the "<<dq[j].v->returnNodecount() << " elements of community " << j << endl; }

	while (current != NULL) {					// insert list elements appropriately
		if (ioparm.textFlag>1) { cout << endl << "element["<<t<<"] from dq["<<j<<"] is ("<<current->x<<" "<<current->y<<")" << endl; }

		// If the element (x,dQ) of [j] is not also (i,dQ) (which it shouldn't be since we've
		// already deleted it previously in this function), and [i] does not also have an
		// element (x,dQ), then we have an ijx-chain.
		if ((current->x != i) && (!dq[i].v->findItem(current->x))) {
			// CASE OF IJX-CHAIN
			
			// First we must calculate the adjustment factor (+).
			double axai = -2.0*a[current->x]*a[i];
			if (ioparm.textFlag>1) {
				cout << "  (0) case of ijx chain: e_{"<<current->x<<" "<<i<<"} absent" << endl;
				cout << "  (1) updating dq["<<current->x<<"] to ("<<j<<" "<<current->y+axai<<")"<<endl;
			}
			
			// Now we must add an element (j,+) to [x], since [x] has essentially now acquired
			// a new connection to [i] (via [j] absorbing [i]).
			dq[current->x].v->insertItem(j, axai);			// (step 1)

			// This new item may have changed the maximum dQ of [x], so we must update it.
			newMax = dq[current->x].v->returnMaxStored();	// (step 3)
			if (ioparm.textFlag>1) { cout << "  (3) dq["<<current->x<<"]'s old maximum was (" << dq[current->x].heap_ptr->m <<" "<< dq[current->x].heap_ptr->j<< ")\t"<<dq[current->x].heap_ptr<<endl; }
			h->updateItem(dq[current->x].heap_ptr, newMax);
			if (ioparm.textFlag>1) {
				cout << "      dq["<<current->x<<"]'s new maximum  is (" << dq[current->x].heap_ptr->m <<" "<< dq[current->x].heap_ptr->j<< ")\t"<<dq[current->x].heap_ptr<<endl;
				cout << "  (4) updating dq["<<j<<"] to ("<<current->x<<" "<<current->y+axai<<")"<<endl;
			}

			// And we must add an element (x,+) to [j], since [j] as acquired a new connection
			// to [x] (via absorbing [i]).
			dq[j].v->insertItem(current->x, axai);			// (step 4)
			newMax = dq[j].v->returnMaxStored();			// (step 6)
			if (ioparm.textFlag>1) { cout << "  (6) dq["<<j<<"]'s old maximum was (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
			h->updateItem(dq[j].heap_ptr, newMax);
			if (ioparm.textFlag>1) { cout << "      dq["<<j<<"]'s new maximum  is (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }

		}    //  (current->x != i && !dq[i].v->findItem(current->x))
		
		temp    = current;
		current = current->next;						// move to next element
		delete temp;
		temp = NULL;
		t++;
	}    // while (current!=NULL)
	
	// Now that we've updated the connections values for all of [i]'s and [j]'s neighbors, 
	// we need to update the a[] vector to reflect the change in fractions of edges after
	// the join operation.
	if (ioparm.textFlag>1) {
		cout << endl;
		cout << "  whoops. no more elements for community "<< j << endl;
		cout << "  (7) updating a["<<j<<"] = " << a[i] + a[j] << " and zeroing out a["<<i<<"]" << endl;
	}
	a[j] += a[i];								// (step 7)
	a[i] = 0.0;
	
	// ---------------------------------------------------------------------------------
	// Finally, now we need to clean up by deleting the vector [i] since we'll never
	// need it again, and it'll conserve memory. For safety, we also set the pointers
	// to be NULL to prevent inadvertent access to the deleted data later on.

	if (ioparm.textFlag>1) { cout << "--> finished merging community "<<i<<" into community "<<j<<" and housekeeping.\n\n"; }
	delete dq[i].v;							// (step 8)
	dq[i].v        = NULL;						// (step 8)
	dq[i].heap_ptr = NULL;						//
	
	return;
 
}

// ------------------------------------------------------------------------------------

bool parseCommandLine(int argc,char * argv[]) {
	int argct = 1;
	string temp, ext;
	string::size_type pos;
	char **endptr;
	long along;
	int count;

	// Description of commandline arguments
	// -f <filename>    give the target .pairs file to be processed
	// -l <text>		the text label for this run; used to build output filenames
	// -t <int>		period of timer for reporting progress of computation to screen
	// -s			calculate and track the support of the dQ matrix
	// -v --v ---v		differing levels of screen output verbosity
	// -c <int>		record the aglomerated network at step <int>
	
	if (argc <= 1) { // if no arguments, return statement about program usage.
		cout << "\nThis program runs the fast community structure inference algorithm due to ";
		cout << "Clauset, Newman and Moore on an input graph in the .pairs format. This version ";
		cout << "is the full max-heap version originally described in cond-mat/0408187. The program ";
		cout << "requires the input network connectivity to be formatted in the following specific ";
		cout << "way: the graph must be simple and connected, where each edge is written on ";
		cout << "a line in the format 'u v w' (e.g., 3481 3483 32).\n";
		cout << "To run the program, you must specify the input network file (-f file.pairs). ";
		cout << "Additionally, you can differentiate runs on the same input file with a label ";
		cout << "(-l test_run) which is imbedded in all corresponding output files. ";
		cout << "Because the algorithm is deterministic, you can specify a point (-c C) at which to ";
		cout << "cut the dendrogram; the program will write out various information about the clustered ";
		cout << "network: a list of clusters, the clustered connectivity, and the cluster size ";
		cout << "distribution. Typically, one wants this value to be the time at which modularity Q ";
		cout << "was maximized (that time is recorded in the .info file for a given run).\n";
		cout << "Examples:\n";
		cout << "  ./FastCommunity -f network.wpairs -l test_run\n";
		cout << "  ./FastCommunity -f network.wpairs -l test_run -c 1232997\n";
		cout << "\n";
		return false;
	}

	while (argct < argc) {
		temp = argv[argct];
		
		if (temp == "-files") {
			cout << "\nBasic files generated:\n";
			cout << "-- .INFO\n";
			cout << "   Various information about the program's running. Includes a listing of ";
			cout << "the files it generates, number of vertices and edges processed, the maximum ";
			cout << "modularity found and the corresponding step (you can re-run the program with ";
			cout << "this value in the -c argument to have it output the contents of the clusters, ";
			cout << "etc. when it reaches that step again (not the most efficient solution, but it ";
			cout << "works)), start/stop time, and when -c is used, it records some information about ";
			cout << "the distribution of cluster sizes.\n";
			cout << "-- .JOINS\n";
			cout << "   The dendrogram and modularity information from the algorithm. The file format ";
			cout << "is tab-delimited columns of data, where the columns are:\n";
			cout << " 1. the community index which absorbs\n";
			cout << " 2. the community index which was absorbed\n";
			cout << " 3. the modularity value Q after the join\n";
			cout << " 4. the time step of the join\n";
			cout << "\nOptional files generated (at time t=C when -c C argument used):\n";
			cout << "-- .WPAIRS\n";
			cout << "   The connectivity of the clustered graph in a .wpairs file format ";
			cout << "(i.e., weighted edges). The edge weights should be the dQ values associated ";
			cout << "with that clustered edge at time C. From this format, it's easy to ";
			cout << "convert into another for visualization (e.g., pajek's .net format).\n";
			cout << "-- .HIST\n";
			cout << "   The size distribution of the clusters.\n";
			cout << "-- .GROUPS\n";
			cout << "   A list of each group and the names of the vertices which compose it (this is ";
			cout << "particularly useful for verifying that the clustering makes sense - tedious but ";
			cout << "important).\n";
			cout << "\n";
			return false;
		} else if (temp == "-f") {			// input file name
			argct++;
			temp = argv[argct];
			ext = ".wpairs";
			pos = temp.find(ext,0);
			if (pos == string::npos) { cout << " Error: Input file must have terminating .wpairs extension.\n"; return false; }
			ext = "/";
			count = 0; pos = string::npos;
			for (int i=0; i < temp.size(); i++) { if (temp[i] == '/') { pos = i; } }
			if (pos == string::npos) {
				ioparm.d_in = "";
				ioparm.filename = temp;
			} else {
				ioparm.d_in = temp.substr(0, pos+1);
				ioparm.filename = temp.substr(pos+1,temp.size()-pos-1);
			}
			ioparm.d_out = ioparm.d_in;
			// now grab the filename sans extension for building outputs files
			for (int i=0; i < ioparm.filename.size(); i++) { if (ioparm.filename[i] == '.') { pos = i; } }
			ioparm.s_scratch = ioparm.filename.substr(0,pos);
		} else if (temp == "-l") {	// s_label
			argct++;
			if (argct < argc) { ioparm.s_label = argv[argct]; }
			else { " Warning: missing modifier for -l argument; using default.\n"; }
			
		} else if (temp == "-t") {	// timer value
			argct++;
			if (argct < argc) {
				along = strtol(argv[argct],endptr,10);
				ioparm.timer = atoi(argv[argct]);
				cout << ioparm.timer << endl;
				if (ioparm.timer == 0 || strlen(argv[argct]) > temp.length()) {
					cout << " Warning: malformed modifier for -t; using default.\n"; argct--;
					ioparm.timer = 20;
				} 
			} else {
				cout << " Warning: missing modifier for -t argument; using default.\n"; argct--;
			}
		} else if (temp == "-c") {	// cut value
			argct++;
			if (argct < argc) {
//				along = strtol(argv[argct],endptr,10);
				ioparm.cutstep = atoi(argv[argct]);
				if (ioparm.cutstep == 0) {
					cout << " Warning: malformed modifier for -c; disabling output.\n"; argct--;
				} 
			} else {
				cout << " Warning: missing modifier for -t argument; using default.\n"; argct--;
			}
		}
		else if (temp == "-s")		{    ioparm.suppFlag = true;		}
		else if (temp == "-v")		{    ioparm.textFlag = 1;		}
		else if (temp == "--v")		{    ioparm.textFlag = 2;		}
		else if (temp == "---v")		{    ioparm.textFlag = 3;		}
		else {  cout << "Unknown commandline argument: " << argv[argct] << endl; }
		argct++;
	}
		
	return true;
}

// ------------------------------------------------------------------------------------

void readInputFile() {
	
	// temporary variables for this function
	int numnodes = 0;
	int numlinks = 0;
	int s,f,t,w;
	edge **last;
	edge *newedge;
	edge *current;								// pointer for checking edge existence
	bool existsFlag;							// flag for edge existence
	time_t t1; t1 = time(&t1);
	time_t t2; t2 = time(&t2);
	
	// First scan through the input file to discover the largest node id. We need to
	// do this so that we can allocate a properly sized array for the sparse matrix
	// representation.
	cout << " scanning input file for basic information." << endl;
	cout << "  edgecount: [0]"<<endl;
	ifstream fscan(ioparm.f_input.c_str(), ios::in);
	while (fscan >> s >> f >> w) {				// read friendship pair (s,f)
		numlinks++;							// count number of edges
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		if (f > numnodes) { numnodes = f; }		// track largest node index

		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2;							// 
			ioparm.timerFlag = true;				// 
		}									// 
		t2=time(&t2);							// 
		
	}
	fscan.close();
	cout << "  edgecount: ["<<numlinks<<"] total (first pass)"<<endl;
	gparm.maxid = numnodes+2;					// store maximum index
	elist = new edge [2*numlinks];				// create requisite number of edges
	int ecounter = 0;							// index of next edge of elist to be used

	// Now that we know numnodes, we can allocate the space for the sparse matrix, and
	// then reparse the file, adding edges as necessary.
	cout << " allocating space for network." << endl;
	e        = new  edge [gparm.maxid];			// (unordered) sparse adjacency matrix
	last     = new edge* [gparm.maxid];			// list of pointers to the last edge in each row
	numnodes = 0;								// numnodes now counts number of actual used node ids
	numlinks = 0;								// numlinks now counts number of bi-directional edges created
	int totweight = 0;							// counts total weight of undirected network
	ioparm.timerFlag = false;					// reset timer
	
	cout << " reparsing the input file to build network data structure." << endl;
	cout << "  edgecount: [0]"<<endl;
	ifstream fin(ioparm.f_input.c_str(), ios::in);
	while (fin >> s >> f >> w) {
		s++; f++;								// increment s,f to prevent using e[0]
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		numlinks++;							// increment link count (preemptive)
		if (e[s].so == 0) {						// if first edge with s, add s and (s,f)
			e[s].so = s;						// 
			e[s].si = f;						// 
			e[s].we = w;						// 
			last[s] = &e[s];					//    point last[s] at self
			numnodes++;						//    increment node count
			totweight += w;					//	 increment weight total
		} else {								//    try to add (s,f) to s-edgelist
			current = &e[s];					// 
			existsFlag = false;					// 
			while (current != NULL) {			// check if (s,f) already in edgelist
				if (current->si==f) {			// 
					existsFlag = true;			//    link already exists
					numlinks--;				//    adjust link-count downward
					break;					// 
				}							// 
				current = current->next;			//    look at next edge
			}								// 
			if (!existsFlag) {					// if not already exists, append it
				newedge = &elist[ecounter++];		//    grab next-free-edge
				newedge -> so = s;				// 
				newedge -> si = f;				// 
				newedge -> we = w;				// 
				totweight += w;				//	 increment weight total
				last[s] -> next = newedge;		//    append newedge to [s]'s list
				last[s]         = newedge;		//    point last[s] to newedge
			}								// 
		}									// 
		
		if (e[f].so == 0) {						// if first edge with f, add f and (f,s)
			e[f].so = f;						// 
			e[f].si = s;						// 
			e[f].we = w;						// 
			last[f] = &e[f];					//    point last[s] at self
			numnodes++;						//    increment node count
			totweight += w;					//	 increment weight total
		} else {								// try to add (f,s) to f-edgelist
			if (!existsFlag) {					//    if (s,f) wasn't in s-edgelist, then
				newedge = &elist[ecounter++];		//       (f,s) not in f-edgelist
				newedge -> so = f;				// 
				newedge -> si = s;				// 
				newedge -> we = w;				// 
				totweight += w;				//	 increment weight total
				last[f] -> next = newedge;		//    append newedge to [f]'s list
				last[f]		 = newedge;		//    point last[f] to newedge
			}								// 
		}									
		existsFlag = false;						// reset existsFlag
		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2;							// 
			ioparm.timerFlag = true;				// 
		}									// 
		t2=time(&t2);							// 
		
	}
	totweight = totweight / 2;					// fix double counting from bi-directed edges
											// (tip to Kimberly Glass for pointing this out)
	cout << "  edgecount: ["<<numlinks<<"] total (second pass)"<<endl;
	cout << "  totweight: ["<<totweight<<"]"<<endl;
	fin.close();
	
	// Now we record our work in the parameters file, and exit.
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---NET_STATS----\n";
	fout << "MAXID-----:\t" << gparm.maxid-2 << "\n";
	fout << "NUMNODES--:\t" << numnodes  << "\n";
	fout << "NUMEDGES--:\t" << numlinks  << "\n";
	fout << "TOTALWT---:\t" << totweight << "\n";
	fout.close();

	gparm.m = numlinks;							// store actual number of edges created
	gparm.n = numnodes;							// store actual number of nodes used
	gparm.w = totweight;						// store actual total weight of edges
	return;
}

// ------------------------------------------------------------------------------------
// records the agglomerated list of indices for each valid community 

void recordGroupLists() {

	list *current;
	ofstream fgroup(ioparm.f_group.c_str(), ios::trunc);
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			fgroup << "GROUP[ "<<i-1<<" ][ "<<c[i].size<<" ]\n";   // external format
			current = c[i].members;
			while (current != NULL) {
				fgroup << current->index-1 << "\n";			// external format
				current = current->next;				
			}
		}
	}
	fgroup.close();
	
	return;
}

// ------------------------------------------------------------------------------------
// records the network as currently agglomerated

void recordNetwork() {

	dpair *list, *current, *temp;
	
	ofstream fnet(ioparm.f_net.c_str(), ios::trunc);
	for (int i=0; i<gparm.maxid; i++) {
		if (dq[i].heap_ptr != NULL) {
			list    = dq[i].v->returnTreeAsList();			// get a list of items in dq[i].v
			current = list;							// store ptr to head of list
			while (current != NULL) {
				//		source		target		weight    (external representation)
				fnet << i-1 << "\t" << current->x-1 << "\t" << current->y << "\n";

				temp = current;						// clean up memory and move to next
				current = current->next;
				delete temp;				
			}
		}		
	}
	fnet.close();
	
	return;
}

// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------




