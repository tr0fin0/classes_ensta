#include "Looper.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


double getTime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <nLoops>\n", argv[0]);

        return 1;
    }

    double nLoops = std::atoi(argv[1]);

    Looper looper;

    double startTime = getTime();
    looper.runLoop(nLoops);
    double endTime = getTime();

    printf("nLoops: %f\n", nLoops);
    printf("Execution time: %f seconds\n", endTime - startTime);

    return 0;
}
