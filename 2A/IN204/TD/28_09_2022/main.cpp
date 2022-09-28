#include <iostream>

#include "Number.hpp"
#include "Array.hpp"
#include "Vector.hpp"


void validation()
{
    std::cout << "validation()" << std::endl << std::endl;

    Number <int> N0;
    Number <int> N1(1);

    N0.print();
    N1.print();
    N1.set(10);
    N1.print();

    std::cout << "C: " << N1.get() << std::endl;



    Array <int, 5> A0;
    A0.fill(10);

    std::cout << "A.size(): " << A0.size() << std::endl;
    std::cout << "A: " << A0[0] << std::endl;
    A0.print();


    Array <float, 5> A1;
    A1.fill(5);

    std::cout << "A.size(): " << A1.size() << std::endl;
    std::cout << "A: " << A1[0] << std::endl;
    A1.print();
};

int main()
{
    std::cout << "TD 28/09/2022" << std::endl << std::endl;

    validation();

    return 0;
};