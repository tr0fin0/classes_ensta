#ifndef INCCOUNTER_HPP
#define INCCOUNTER_HPP
#include "counter.hpp"
 
class incCounter : public counter
{

public:
    void inc() { i++; }
 
protected:
    incCounter(): counter() {}
    incCounter(unsigned counter): counter(counter) {}
    incCounter(const counter& otherCounter): counter(otherCounter.i) {}
    ~incCounter() {}

};

#endif // INCCOUNTER_HPP