#include "timespec.h"

// conversion
timespec timespec_normalize(timespec& time_ts){
    if (time_ts.tv_nsec >= 1000000000L) {
        time_ts.tv_sec += time_ts.tv_nsec / 1000000000L;
        time_ts.tv_nsec %= 1000000000L;

    } else if (time_ts.tv_nsec < 0) {
        time_ts.tv_sec -= 1 + (-time_ts.tv_nsec / 1000000000L);
        time_ts.tv_nsec = 1000000000L - (-time_ts.tv_nsec % 1000000000L);
    }

    return time_ts;
}
double timespec_to_ms(const timespec& time_ts){
    return time_ts.tv_sec * 1000.0 + time_ts.tv_nsec / 1000000.0;
}
timespec timespec_from_ms(double time_ms){
    timespec ts;
    ts.tv_sec = static_cast<time_t>(time_ms / 1000);
    ts.tv_nsec = static_cast<long>((time_ms - ts.tv_sec * 1000) * 1000000);

    return timespec_normalize(ts);
}

timespec timespec_now(){
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    return ts;
}
timespec timespec_negate(const timespec& time_ts){
    timespec ts = {-time_ts.tv_sec, -time_ts.tv_nsec};

    return timespec_normalize(ts);
}

timespec timespec_add(const timespec& time1_ts, const timespec& time2_ts){
    timespec result = {time1_ts.tv_sec + time2_ts.tv_sec, time1_ts.tv_nsec + time2_ts.tv_nsec};

    return timespec_normalize(result);
}
timespec timespec_subtract(const timespec& time1_ts, const timespec& time2_ts){
    timespec result = {time1_ts.tv_sec - time2_ts.tv_sec, time1_ts.tv_nsec - time2_ts.tv_nsec};

    return timespec_normalize(result);
}

void timespec_wait(const timespec& delay_ts){
    std::this_thread::sleep_for(std::chrono::seconds(delay_ts.tv_sec) + std::chrono::nanoseconds(delay_ts.tv_nsec));
}

timespec  operator- (const timespec& time_ts){
    return timespec_negate(time_ts);
}
timespec  operator+ (const timespec& time1_ts, const timespec& time2_ts){
    return timespec_add(time1_ts, time2_ts);
}
timespec  operator- (const timespec& time1_ts, const timespec& time2_ts){
    return timespec_subtract(time1_ts, time2_ts);
}
timespec& operator+= (timespec& time_ts, const timespec& delay_ts){
    time_ts = time_ts + delay_ts;

    return time_ts;
}
timespec& operator-= (timespec& time_ts, const timespec& delay_ts){
    time_ts = time_ts - delay_ts;

    return time_ts;
}
bool operator== (const timespec& time1_ts, const timespec& time2_ts){
    return (time1_ts.tv_sec == time2_ts.tv_sec) && (time1_ts.tv_nsec == time2_ts.tv_nsec);
}
bool operator!= (const timespec& time1_ts, const timespec& time2_ts){
    return !(time1_ts == time2_ts);
}
bool operator< (const timespec& time1_ts, const timespec& time2_ts){
    return (time1_ts.tv_sec < time2_ts.tv_sec) || 
           (time1_ts.tv_sec == time2_ts.tv_sec && time1_ts.tv_nsec < time2_ts.tv_nsec);
}
bool operator> (const timespec& time1_ts, const timespec& time2_ts){
    return time2_ts < time1_ts;
}