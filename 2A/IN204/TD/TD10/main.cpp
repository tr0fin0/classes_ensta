#include <thread>
#include <iostream>

int var = 0;

int max(int n, int m) {
    if (n > m) {
        return n;
    }
    else {
        return m;
    };
}


void enumerate(int n) {
    std::cout << "n = " << n << ": ";

    int i = 0;
    for (i; i < max(var, n); i++) {
        std::cout << i << " ";
    }
    var += i;

    std::cout << "(n, v) = (" << n << ", " << var << ")" << std::endl;
};


int main() {
    std::cout << "TD 15/11/2022" << std::endl;

    std::thread th1(enumerate, 7);
    std::thread th2(enumerate, 6);
    std::thread th3(enumerate, 5);
    th1.join();
    th2.join();
    th3.join();

    std::cout << var;
    return 0;
}