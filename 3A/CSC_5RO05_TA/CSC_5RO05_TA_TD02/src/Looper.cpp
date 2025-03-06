#include "Looper.h"

Looper::Looper() : doStop(false), iLoop(0) {}

double Looper::runLoop(double nLoops)
{
    iLoop = 0;
    doStop = false;

    while (iLoop < nLoops && !doStop)
    {
        iLoop++;
    }

    return iLoop;
}

void Looper::stopLoop()
{
    doStop = true;
}

double Looper::getSample() const
{
    return iLoop;
}
