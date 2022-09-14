#include <iostream>
#include "Counter.h"
#include "Double.h"
using namespace std;


int main() {
    cout << "class definition" << endl;

    Counter count(0);
    count.print();
    count.inc();
    count.print();


    Double dcount(0);
    dcount.print();
    dcount.doubleInc();
    dcount.print();


    return 0;
}