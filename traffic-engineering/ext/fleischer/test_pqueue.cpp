#include "pqueue.h"
#include <float.h>
#include <iostream>
#include <map>

using namespace std;


int main()
{
	cout << "testing pqueue impl" << endl;

	int *positions = new int[10]; // assuming there are ten entries
	map<int, double> items = {
		{0, 1.0}, {1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 10}, {5, 6}, {6, 0.01}, {7, 0.8}, {8, .2}, {9, 100}
	};		

	PQUEUE pq(10);
	
	PQDATUM item;
	map<int, double>::iterator midi;
	for(midi = items.begin(); midi != items.end(); midi++)
	{
		item.set_node((*midi).first);
		item.set_dist((*midi).second);

		pq.pqinsert(item);
	}
	

	cout << pq.size() << " " << pq.avail() << endl;

	
	while(pq.size() > 0)
	{
		pq.pqremove(&item);
		cout << "minel = " << item.node() << " " << item.dist() << endl;
	}	
}
