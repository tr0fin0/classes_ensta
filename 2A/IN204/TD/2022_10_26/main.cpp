#include <iostream>
#include "counter.hpp"
// #include "decCounter.hpp"
// #include "incCounter.hpp"
// #include "allCounter.hpp"

int main() {
    std::cout << "TD 07/09/2022\n";
    // in this TD we wish to solve the diamond problem inheritance:
    //   A
    //  / \
    // B   C
    //  \ /
    //   D

    // in theory this is what would happen when defining the classes but, without:
    // virtual <public, protected, private>

    // the compiler would understand:
    // A   A
    // |   |
    // B   C
    //  \ /
    //   D

    counter A();
    A.get();
    A.set(10);
    A.get();

    // decCounter B();
    // incCounter C();
    // allCounter D();



    return 0;
}