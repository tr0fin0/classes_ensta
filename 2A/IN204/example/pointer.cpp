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