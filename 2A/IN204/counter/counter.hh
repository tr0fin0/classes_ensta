#include <stdio.h>

struct counter
{
    int i;

    counter(int value)
    {
        this -> i = value;
    };

    void print() const // const does not allow to change the function in the object
    {
        printf("%d\n", i);
    };

    void inc()
    {
        this -> i++;
    };

};