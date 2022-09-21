#include <iostream>
#include "join.h"
using namespace std;


void validation()
{
    printf("%d\n", joinDigits(123, 4));
    printf("%d\n", joinDigitsGeneric(123, 4));
    printf("%f\n", joinDigitsGeneric(12.30, 4.0));
    printf("%f\n", joinDigitsGeneric(12.34, 5.0));
    printf("%f\n", joinDigitsGeneric(123.0, 4.0));
    printf("%f\n", joinDigitsGeneric(123.0, 0.4));

};

int main() {
    cout << "concatenate" << endl << endl;


    validation();


    return 0;
}