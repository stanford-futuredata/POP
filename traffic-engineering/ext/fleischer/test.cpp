#include <fstream>
#include <iostream>
using namespace std;

int main()
{
  char data[100];

  ifstream input("y.txt");
  input.getline(data, 100);

  // cin.getline(data, 100);
  cout << data << endl;
}
