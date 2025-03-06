#include "CountDown.h"
#include <stdio.h>

int main()
{
    CountDown cd(10);
    timespec duration = {1, 0};
    cd.start(duration, true);

    while (cd.count > 0);

    return 0;
}