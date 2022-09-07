#include <iostream>
#include "counter.hh"

void run_inc(counter &count)
{
    std::cout << "x = " << count.i << "\n";

    count.inc();
    std::cout << "x = " << count.i << "\n";
};

void run_print(const counter &count)
{
    count.print();
};


int main() {
    std::cout << "main.cpp\n";

    counter count(8);

    run_inc(count);
    run_print(count);


    return 0;
}
