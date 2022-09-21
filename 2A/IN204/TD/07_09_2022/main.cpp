#include <iostream>
#include "counter.hpp"

void validation(counter &count)
{
    std::cout << "x = " << count.i << "\n";

    count.inc();
    std::cout << "x = " << count.i << "\n";

    count.print();
};


int main() {
    std::cout << "TD 07/09/2022\n";

    counter count(8);
    validation(count);


    return 0;
}
