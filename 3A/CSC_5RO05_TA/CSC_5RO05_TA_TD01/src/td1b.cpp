#include "Chrono.h"
#include <stdio.h>



void debug(const char* label, timespec ts)
{
    printf("%s:\t%ld s, %ld ns\n", label, ts.tv_sec, ts.tv_nsec);
}


int main(void)
{
    // Chrono
    Chrono chrono;
    printf("Chronometer started.\n");
    timespec_wait(timespec_from_ms(500));

    // Lap
    timespec lap_time = chrono.lap();
    debug("Lap time", lap_time);
    printf("Lap time in ms: %.3f ms\n", chrono.lap_ms());
    timespec_wait(timespec_from_ms(1000));

    // Stop
    timespec stop_time = chrono.stop();
    debug("Stop time", stop_time);
    printf("Stopped. Total elapsed time: %.3f ms\n", chrono.lap_ms());

    // Restart
    chrono.restart();
    printf("Chronometer restarted.\n");
    timespec_wait(timespec_from_ms(700));

    printf("Final lap time after restart: %.3f ms\n", chrono.lap_ms());

    return 0;
}
