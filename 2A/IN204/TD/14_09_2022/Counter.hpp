#include <iostream>
using namespace std;

class Counter
{
protected:                  // protected variables and methods


public:                     // public variables and methods
    int i = 0;

    Counter() {};           // constructor defaut
    Counter(int value)      // constructor
    {
        this -> i = value;
    };

    ~Counter() {};          //destructor

    void print() const // const does not allow to change the function in the object
    {
        printf("%d\n", i);
    };

    int inc()
    {
        return this -> i++;
    };

private:                    // private variables and methods

};