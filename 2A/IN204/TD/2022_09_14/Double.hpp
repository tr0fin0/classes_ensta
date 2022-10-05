#include <iostream>
// #include "Counter.h"
using namespace std;

class Double : public Counter
{
    using Counter::Counter;

public:
    Double() : Counter() {};
    Double(int i) : Counter(i) {};
    ~Double() {};

    int doubleInc()
    {
        return this -> inc() + this -> inc();
    };

};