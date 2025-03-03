#include "Timer.h"

Timer::Timer()
{
    this->tid = 0;
    this->isRunning = false;
};

Timer::~Timer()
{
    if (this->isRunning)
    {
        stop();
    }
};

void::Timer::start(timespec duration, bool isPeriodic)
{
    if (this->isRunning)
    {
        stop();
    }

    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = call_callback;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGRTMIN, &sa, nullptr);


    struct sigevent sev;
    struct itimerspec its{{0, 0}, {0, 0}};

    sev.sigev_notify = SIGEV_THREAD;
    sev.sigev_value.sival_ptr = &this->tid;


    timer_create(CLOCK_REALTIME, &sev, &this->tid);

    its.it_value = duration;
    its.it_interval = isPeriodic ? duration : timespec{0, 0};

    timer_settime(this->tid, 0, &its, NULL);

    this->isRunning = true;
};

void::Timer::start_ms(double duration_ms, bool isPeriodic)
{
    timespec duration = timespec_from_ms(duration_ms);

    start(duration, isPeriodic);
};

void::Timer::stop()
{
    if (this->isRunning)
    {
        timer_delete(this->tid);
        this->isRunning = false;
    }
};

void::Timer::call_callback(int, siginfo_t* si, void*)
{
    Timer* timer = static_cast<Timer*>(si->si_value.sival_ptr);

    timer->callback();
};
