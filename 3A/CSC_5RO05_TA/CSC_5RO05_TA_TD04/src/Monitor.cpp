#include "Monitor.h"


Monitor::Monitor(Mutex& mutex) : mutex(mutex)
{
    pthread_cond_init(&posixCondId, nullptr);
}

Monitor::~Monitor()
{
    pthread_cond_destroy(&posixCondId);
}

void Monitor::notify()
{
    pthread_cond_signal(&posixCondId);
}

void Monitor::notifyAll()
{
    pthread_cond_broadcast(&posixCondId);
}


Monitor::Lock::Lock(Monitor& monitor) : monitor(monitor)
{
    pthread_mutex_lock(&monitor.mutex.posixMutexId);
}

Monitor::Lock::Lock(Monitor& monitor, long timeout_ms) : monitor(monitor)
{
    timespec ts = timespec_now() + timespec_from_ms(timeout_ms);

    pthread_mutex_timedlock(&monitor.mutex.posixMutexId, &ts);
}

Monitor::Lock::~Lock()
{
    pthread_mutex_unlock(&monitor.mutex.posixMutexId);
}

bool Monitor::Lock::wait()
{
    return pthread_cond_wait(&monitor.posixCondId, &monitor.mutex.posixMutexId) == 0;
}

bool Monitor::Lock::wait(long timeout_ms)
{
    timespec ts = timespec_now() + timespec_from_ms(timeout_ms);

    return pthread_cond_timedwait(&monitor.posixCondId, &monitor.mutex.posixMutexId, &ts) == 0;
}
