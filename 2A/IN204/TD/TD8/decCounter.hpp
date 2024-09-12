#ifndef DECCOUNTER_HPP
#define DECCOUNTER_HPP
#include "counter.hpp"
 
class decCounter : virtual public counter
{

public:
    void dec() { i--; }
 
protected:
    decCounter(): counter() {}
    decCounter(unsigned counter): counter(counter) {}
    decCounter(const counter& otherCounter): counter(otherCounter.i) {}
    ~decCounter() {}

};

#endif // DECCOUNTER_HPP