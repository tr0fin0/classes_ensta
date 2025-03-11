#ifndef TIMESPEC_H
#define TIMESPEC_H

#include <time.h>
#include <thread>

// conversion
double timespec_to_ms(const timespec& time_ts);
timespec timespec_from_ms(double time_ms);
timespec timespec_normalize(const timespec& time_ts);

timespec timespec_now();
timespec timespec_negate(const timespec& time_ts);

timespec timespec_add(const timespec& time1_ts, const timespec& time2_ts);
timespec timespec_subtract(const timespec& time1_ts, const timespec& time2_ts);

void timespec_wait(const timespec& delay_ts);

timespec  operator- (const timespec& time_ts);
timespec  operator+ (const timespec& time1_ts, const timespec& time2_ts);
timespec  operator- (const timespec& time1_ts, const timespec& time2_ts);
timespec& operator+= (timespec& time_ts, const timespec& delay_ts);
timespec& operator-= (timespec& time_ts, const timespec& delay_ts);
bool operator== (const timespec& time1_ts, const timespec& time2_ts);
bool operator!= (const timespec& time1_ts, const timespec& time2_ts);
bool operator< (const timespec& time1_ts, const timespec& time2_ts);
bool operator> (const timespec& time1_ts, const timespec& time2_ts);

#endif // TIMESPEC_H
