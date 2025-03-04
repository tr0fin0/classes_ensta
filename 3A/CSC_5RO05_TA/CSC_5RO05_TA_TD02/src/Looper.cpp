#include "Looper.h"

Looper::Looper() : doStop(false), iLoop(0), nLoops(0) {}

void Looper::callback() {
    if (iLoop < nLoops && !doStop) {
        iLoop += 1.0;  // Increment loop counter
    } else {
        stop();  // Stop the timer when done
    }
}

double Looper::runLoop(double loops) {
    iLoop = 0;
    doStop = false;
    nLoops = loops;

    start_ms(1, true);  // Start the timer with a 1 ms period

    while (iLoop < nLoops && !doStop) {
        // Busy wait (not ideal, but avoids threading)
    }

    stop();  // Ensure timer stops after completion
    return iLoop;
}

void Looper::stopLoop() {
    doStop = true;  // Ensures loop stops
}

double Looper::getSample() const {
    return iLoop;
}
