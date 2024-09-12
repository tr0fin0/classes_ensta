#ifndef ALLCOUNTER_HPP
#define ALLCOUNTER_HPP
#include "incCounter.hpp"
#include "decCounter.hpp"
 
class allCounter : virtual public decCounter, incCounter
{

public:
 
protected:
    allCounter(): 
        decCounter(), 
        incCounter() {}
    allCounter(unsigned counter): 
        decCounter(counter),
        incCounter(counter) {}
    allCounter(const counter& otherCounter): 
        decCounter(otherCounter.i), 
        incCounter(otherCounter.i) {}
    ~allCounter() {}

};

#endif // INCCOUNTER_HPP