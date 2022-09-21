#include <iostream>
#include "Counter.hpp"
#include "Double.hpp"
using namespace std;


int main() {
    cout << "TD 14/09/2022" << endl;

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