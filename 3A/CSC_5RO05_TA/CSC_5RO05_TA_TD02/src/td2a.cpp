#include <time.h>
#include <signal.h>
#include <stdio.h>



void timer_handler(int sig, siginfo_t* si, void*)
{
    (void) sig;

    auto p_counter = (int*) si->si_value.sival_ptr;

    *p_counter += 1;
    printf("counter: %d\n", *p_counter);
}


int main(void)
{
    int counter = 0;

    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = timer_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGRTMIN, &sa, nullptr);

    struct sigevent sev;
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = SIGRTMIN;
    sev.sigev_value.sival_ptr = (void*) &counter;

    timer_t tid;
    timer_create(CLOCK_REALTIME, &sev, &tid);
    itimerspec its;

    // usually first and following iterations will be the same
    its.it_value = timespec{0, 500000000};
    its.it_interval = timespec{0, 500000000};

    timer_settime(tid, 0, &its, nullptr);

    while (counter < 15) {
    }

    timer_delete(tid);

    return 0;
}