#include "Calibrator.h"
#include "Chrono.h"
// #include "timespec.h"
// #include <iostream>
// #include <cstdlib>
// #include <ctime>


// double getTime()
// {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);

//     return ts.tv_sec + ts.tv_nsec * 1e-9;
// }

int main() {
    // double samplingPeriod_ms = 200;
    // unsigned nSamples = 10;
    // printf("main\n");
    // Calibrator calibrator(samplingPeriod_ms, nSamples);

    // Looper looper;

    // double startTime = getTime();
    // looper.runLoop(calibrator.nLoops(100));
    // double endTime = getTime();

    // printf("nLoops: %f\n", calibrator.nLoops(100));
    // printf("Execution time: %f seconds\n", endTime - startTime);


    double samplingPeriod_ms = 200;
    unsigned nSamples = 10;
    Calibrator calibrator(samplingPeriod_ms, nSamples);

    Looper looper;

    Chrono chrono;
    looper.runLoop(calibrator.nLoops(100));
    double executionTime = timespec_to_ms(chrono.stop());

    printf("nLoops: %f\n", calibrator.nLoops(100));
    printf("Execution time: %f seconds\n", executionTime);

    return 0;
}