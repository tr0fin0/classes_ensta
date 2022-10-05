#include <iostream>
using namespace std;

int main() {
    cout << "pointer arithmetic" << endl << endl;

    int var = 5;
    int* pointVar;
    pointVar = &var;

    cout << "variable" << endl;
    cout << "value:    var = " <<  var << endl;             //  var = 5
    cout << "address: &var = " << &var << endl << endl;     // &var = 0x61fe14


    cout << "pointer" << endl;
    cout << "address, pointVar = " <<  pointVar << endl;    //  pointVar = 0x61fe14
    cout << "value,  *pointVar = " << *pointVar << endl;    // *pointVar = 5

    return 0;