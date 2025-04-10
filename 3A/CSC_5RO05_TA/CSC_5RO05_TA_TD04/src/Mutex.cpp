#include "Mutex.h"

Mutex::Mutex()
{
    pthread_mutex_init(&posixMutexId, nullptr);
}

Mutex::~Mutex()
{
    pthread_mutex_destroy(&posixMutexId);
}

void Mutex::lock()
{
    pthread_mutex_lock(&posixMutexId);
}

bool Mutex::lock(double timeout_ms)
{
    timespec ts = timespec_now() + timespec_from_ms(timeout_ms);

    return pthread_mutex_timedlock(&posixMutexId, &ts) == 0;
}

void Mutex::unlock()
{
    pthread_mutex_unlock(&posixMutexId);
}


Mutex::Lock::Lock(Mutex& mutex) : mutex(mutex)
{
    mutex.lock();
}

Mutex::Lock::Lock(Mutex& mutex, double timeout_ms) : mutex(mutex)
{
    if (not mutex.lock(timeout_ms))
        throw TimeoutException(timeout_ms);
}

Mutex::Lock::~Lock()
{
    mutex.unlock();
}
