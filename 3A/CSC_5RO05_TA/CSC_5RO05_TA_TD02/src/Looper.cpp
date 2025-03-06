#include "Looper.h"

Looper::Looper() : doStop(false), iLoop(0), nLoops(0) {}

void Looper::callback()
{
    if (iLoop < nLoops && !doStop)
    {
        iLoop += 1.0;
    }
    else
    {
        stop();
    }
}

double Looper::runLoop(double loops)
{
    iLoop = 0;
    doStop = false;
    nLoops = loops;

    start_ms(1, true);

    while (iLoop < nLoops && !doStop) {}

    stop();
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
