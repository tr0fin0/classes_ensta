#ifndef MUTEX_H
#define MUTEX_H

#include <mutex>
#include "TimeoutException.h"
#include "timespec.h"

class Mutex
{
public:
    class Lock;

private:
    pthread_mutex_t posixMutexId;

public:
    Mutex();
    ~Mutex();

private:
    void lock();
    bool lock(double timeout_ms);
    void unlock();
};


class Mutex::Lock
{
private:
    Mutex& mutex;

public:
    Lock(Mutex& mutex);
    Lock(Mutex& mutex, double timeout_ms);
    ~Lock();
};


#endif // MUTEX_H