#include "Monitor.h"
#include "Mutex.h"


class Semaphore
{
public:
    using CountType = unsigned long;

private:
    Monitor& monitor;
    Mutex& mutex;

    CountType counter;
    CountType maxCount;

public:
    Semaphore(Monitor& monitor, Mutex& mutex, CountType initValue, CountType maxValue);
    ~Semaphore();
    void give();
    bool take();
    bool take(long timeout_ms);
};
