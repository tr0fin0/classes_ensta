// CpuLoop.cpp
#include "CpuLoop.h"
#include "Chrono.h"

CpuLoop::CpuLoop(Calibrator& calib) : calibrator(calib) {}

double CpuLoop::runTime(double duration_ms) {
    double nLoops = calibrator.nLoops(duration_ms);

    Chrono chrono;
    runLoop(nLoops);
    chrono.stop();
    double actualDuration = timespec_to_ms(chrono.lap());
    return (actualDuration - duration_ms) / duration_ms;
}