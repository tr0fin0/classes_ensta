// CpuLoop.h
#ifndef CPULOOP_H
#define CPULOOP_H

#include "Looper.h"
#include "Calibrator.h"

class CpuLoop : public Looper {
private:
    Calibrator& calibrator;

public:
    CpuLoop(Calibrator& calib);
    double runTime(double duration_ms);
};

#endif // CPULOOP_H
