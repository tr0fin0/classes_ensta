#include "Timer.h"

Timer::Timer()
{
    tid = 0;
    isRunning = false;

    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = call_callback;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGRTMIN, &sa, nullptr);

    struct sigevent sev;
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = SIGRTMIN;
    sev.sigev_value.sival_ptr = this;

    timer_create(CLOCK_REALTIME, &sev, &tid);
};

Timer::~Timer()
{
    if (isRunning)
    {
        stop();
    }
};

void::Timer::start(timespec duration, bool isPeriodic)
{
    if (isRunning)
    {
        stop();
    }

    struct itimerspec its;
    its.it_value = duration;
    its.it_interval = isPeriodic ? duration : timespec{0, 0};

    timer_settime(tid, 0, &its, NULL);

    isRunning = true;
};

void::Timer::start_ms(double duration_ms, bool isPeriodic)
{
    timespec duration = timespec_from_ms(duration_ms);

    start(duration, isPeriodic);
};

void::Timer::stop()
{
    if (isRunning)
    {
        timer_delete(tid);

        isRunning = false;
    }
};

void::Timer::call_callback(int, siginfo_t* si, void*)
{
    Timer* timer = static_cast<Timer*>(si->si_value.sival_ptr);

    timer->callback();
};
