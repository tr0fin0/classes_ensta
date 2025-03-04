#ifndef TIMER_H
#define TIMER_H

#include "timespec.h"
#include <signal.h>
#include <stdio.h>

class Timer
{
private:
    timer_t tid;
    bool isRunning;

private:
    static void call_callback(int, siginfo_t* si, void*);

protected:
    virtual void callback() = 0;

public:
    Timer();
    virtual ~Timer();
    void start(timespec duration, bool isPeriodic);
    void start_ms(double duration_ms, bool isPeriodic);
    void stop();
};

#endif // TIMER_H
