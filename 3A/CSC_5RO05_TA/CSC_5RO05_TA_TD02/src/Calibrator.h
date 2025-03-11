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
protected:
    void callback() override;

public:
    Calibrator(double samplingPeriod_ms, unsigned nSamples);
    double nLoops(double duration_ms);
};

#endif // CALIBRATOR_H
