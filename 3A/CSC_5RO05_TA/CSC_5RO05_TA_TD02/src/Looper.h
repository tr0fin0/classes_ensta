#ifndef LOOPER_H
#define LOOPER_H

#include "Timer.h"

class Looper : public Timer
{
private:
    volatile bool doStop;
    volatile double iLoop;
    double nLoops;

protected:
    void callback() override;

public:
    Looper();
    double runLoop(double nLoops);
    void stopLoop();
    double getSample() const;
};

#endif // LOOPER_H
