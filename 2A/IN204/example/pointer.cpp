#include <iostream>
using namespace std;

int main() {
    cout << "pointer arithmetic" << endl << endl;

    int var = 5;
    int* pointVar;
    pointVar = &var;

    cout << "variable" << endl;
    cout << "value:    var = " <<  var << endl;
    cout << "address: &var = " << &var << endl << endl;


    cout << "pointer" << endl;
    cout << "address, pointVar = " <<  pointVar << endl;
    cout << "value,  *pointVar = " << *pointVar << endl;

    return 0;
}

// OUTPUT________________________
// pointer arithmetic

// variable
// value:    var = 5
// address: &var = 0x61fe14

// pointer
// address, pointVar = 0x61fe14
// value,  *pointVar = 5