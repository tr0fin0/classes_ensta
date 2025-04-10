#include "Semaphore.h"
#include <stdexcept>


Semaphore::Semaphore(Monitor& monitor, Mutex& mutex, CountType initValue, CountType maxValue) :
    monitor(monitor),
    mutex(mutex),
    counter(initValue),
    maxCount(maxValue)
{
    if (initValue < 0)
        throw std::invalid_argument("initValue cannot be negative.");

    if (maxValue < 0)
        throw std::invalid_argument("maxValue cannot be negative.");

    if (initValue > maxValue)
        throw std::invalid_argument("initValue cannot be greater than maxValue.");
};

Semaphore::~Semaphore()
{

}

void Semaphore::give()
{
    Monitor::Lock lock(monitor);

    if (counter < maxCount)
    {
        counter++;
        monitor.notify();
    }
}

bool Semaphore::take()
{
    Monitor::Lock lock(monitor);

    while (counter == 0)
        lock.wait();

    counter--;
    return true;
}

bool Semaphore::take(long timeout_ms)
{
    Monitor::Lock lock(monitor);
    timespec startTime = timespec_now();
    timespec timeout = timespec_from_ms(timeout_ms);

    while (counter == 0)
    {
        timespec remainingTime = timeout - (timespec_now() - startTime);
        if (remainingTime < timespec_from_ms(0))
            return false;

        if (not lock.wait(timespec_to_ms(remainingTime)))
            return false;
    }

    counter--;
    return true;
}
