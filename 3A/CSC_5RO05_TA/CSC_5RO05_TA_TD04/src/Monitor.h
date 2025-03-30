#ifndef MONITOR_H
#define MONITOR_H

#include "Mutex.h"

class Monitor
{
private:
    pthread_cond_t posixCondId;
    Mutex& mutex;

public:
    class Lock;


public:
    Monitor(Mutex& mutex);
    ~Monitor();
    void notify();
    void notifyAll();
};

class Monitor::Lock
{
private:
    Monitor& monitor;


public:
    Lock(Monitor& monitor);
    Lock(Monitor& monitor, long timeout_ms);
    ~Lock();
    bool wait();
    bool wait(long timeout_ms);
};

#endif  // MONITOR_H