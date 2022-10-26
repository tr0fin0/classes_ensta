#ifndef COUNTER_HPP
#define COUNTER_HPP
 
class counter
{
public:
    unsigned i;
 
public:
    unsigned get() const { return i; }
    void reset() { i = 0; }
    void set(unsigned value) { i = value; }
 
protected:
    counter(): i(0) {}
    counter(unsigned counter): i(counter) {}
    counter(const counter& otherCounter): counter(otherCounter.i) {}
    ~counter() {}
};
 
#endif // COUNTER_HPP