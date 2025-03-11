#include "Calibrator.h"

#include <climits>
#include "Looper.h"
#include "Timer.h"

Calibrator::Calibrator(double samplingPeriod_ms, unsigned nSamples)
    : a(0), b(0), iterations(nSamples), samples(), looper()
{
    start_ms(samplingPeriod_ms, true);
    looper.runLoop(__DBL_MAX__);

    printf("linear regression\n");
    if (samples.size() > 0) {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (unsigned i = 0; i < samples.size(); ++i) {
            double x = i * samplingPeriod_ms;
            double y = samples[i];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }

        double n = samples.size();
        a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        b = (sumY - a * sumX) / n;
    }
}

void Calibrator::callback()
{
    samples.push_back(looper.getSample());

    if (samples.size() >= iterations)
    {
        looper.stopLoop();
        stop();
    }
    printf("samples.size()=%ld\n", samples.size());
}

double Calibrator::nLoops(double duration_ms)
{
    return a * duration_ms + b;
}


