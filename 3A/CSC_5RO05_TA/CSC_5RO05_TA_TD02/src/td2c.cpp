#include "Looper.h"
#include <iostream>
#include <cstdlib>   // For atoi()
#include <ctime>     // For clock_gettime()

// Function to get time in seconds
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <nLoops>" << std::endl;
        return 1;
    }

    double nLoops = std::atoi(argv[1]);  // Convert input to integer

    Looper looper;
    
    double startTime = getTime();  // Start timing
    looper.runLoop(nLoops);
    double endTime = getTime();    // End timing

    std::cout << "nLoops: " << nLoops << std::endl;
    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}
