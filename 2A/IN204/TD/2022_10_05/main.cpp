#include <iostream>
#include "Number.hpp"


int main()
{
    std::cout << "TD 05/10/2022" << std::endl << std::endl;

    Number<float> N0(8);
    Number<float> N1(8);


    std::cout << N0.get() << std::endl;
    std::cout << N1.get() << std::endl << std::endl;

    // N0 = N1 + N1 + N1;
    N0 = N1 + 1;
    std::cout << N0.get() << std::endl;

    // N0 = N1 - N1 - N1;
    N0 = N1 - 1;
    std::cout << N0.get() << std::endl;

    // N0 = N1 * N1 * N1;
    N0 = N1 * 1;
    std::cout << N0.get() << std::endl;

    // N0 = N1 / N1 / N1;
    N0 = N1 / 1;
    std::cout << N0.get() << std::endl;
    std::cout << "cout: " << N0 << std::endl;


    return 0;
};