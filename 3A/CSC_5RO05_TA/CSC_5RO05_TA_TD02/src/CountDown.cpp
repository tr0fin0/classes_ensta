#include "CountDown.h"

CountDown::CountDown(int n) : count(n) {}

void CountDown::callback()
{
    if (count > 0)
    {
        printf("CountDown: %d\n", count);
        count--;
    }
    else
    {
        stop();
    }
}
