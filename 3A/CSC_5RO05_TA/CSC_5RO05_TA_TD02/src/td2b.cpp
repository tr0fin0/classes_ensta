#include "Timer.h"
#include <stdio.h>

class CountDown : public Timer
{
private:
    int count;

public:
    CountDown(int n) : count(n) {}

    void callback() override
    {
        if (count >= 0)
        {
            printf("counter: %d\n", count);
        }
        else
        {
            stop();
        }
    }
};

int main() {
    CountDown cd(10);
    timespec duration = {1, 0};
    cd.start(duration, true);

    while (true)
    {
        sleep(1);
    }

    return 0;
}