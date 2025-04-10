#include "CpuLoop.h"
#include "Calibrator.h"
#include "Chrono.h"
#include <cstdio>

int main() {
    double samplingPeriod_ms = 200.0;
    unsigned int nSamples = 10;
    Calibrator calibrator(samplingPeriod_ms, nSamples);
    CpuLoop cpuLoop(calibrator);

    printf("Requested Duration (s) | Measured Duration (s) | Relative Error (%%)\n");

    for (double duration = 0.5; duration <= 10.0; duration += 0.2) {
        Chrono chrono;
        chrono.start();
        double error = cpuLoop.runTime(duration * 1000); // Convert to milliseconds
        chrono.stop();

        double measuredDuration = chrono.lap() / 1000.0; // Convert to seconds
        printf("%.2f | %.2f | %.2f%%\n", duration, measuredDuration, error * 100);
    }
    return 0;
}