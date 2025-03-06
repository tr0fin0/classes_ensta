#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "Timer.h"
#include <vector>

class Calibrator : public Timer
{
private:
    double a;
    double b;
    std::vector<double> samples;

public:
    Calibrator(double samplingPeriod_ms, unsigned nSamples);
    void callback() override;
    double nLoops(double duration_ms);
};

#endif // CALIBRATOR_H
