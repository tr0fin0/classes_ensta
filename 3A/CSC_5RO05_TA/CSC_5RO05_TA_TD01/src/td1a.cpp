#include "timespec.h"
#include <stdio.h>



void debug(const char* label, timespec ts)
{
    printf("%s:\t%ld s, %ld ns\n", label, ts.tv_sec, ts.tv_nsec);
}


int main(void)
{
    timespec now = timespec_now();
    debug("timespec_now()", now);
    printf("\n\n");


    // Conversion
    double reference = 1234.567;
    timespec ts_reference = timespec_from_ms(reference);
    printf("Reference [ms]: ref = %.3f\n", timespec_to_ms(ts_reference));
    debug("timespec_from_ms()", ts_reference);
    printf("\n\n");

    timespec negated = timespec_negate(ts_reference);
    debug("timespec_negate(ref)", negated);

    timespec sum = timespec_add(ts_reference, ts_reference);
    debug("timespec_add(ref, ref)", sum);

    timespec diff = timespec_subtract(ts_reference, ts_reference);
    debug("timespec_sub(ref, ref)", diff);
    printf("\n\n");


    // Operators
    timespec op_neg = -ts_reference;
    debug("operator !", op_neg);

    timespec op_sum = ts_reference + ts_reference;
    debug("operator +", op_sum);

    timespec op_diff = ts_reference - ts_reference;
    debug("operator -", op_diff);

    ts_reference += ts_reference;
    debug("operator +=", ts_reference);

    ts_reference -= ts_reference;
    debug("operator -=", ts_reference);

    printf("operator ==:\t%d\n", ts_reference == ts_reference);
    printf("operator !=:\t%d\n", ts_reference != ts_reference);
    printf("operator <: \t%d\n", ts_reference < now);
    printf("operator >: \t%d\n", now > ts_reference);
    printf("\n\n");


    // Test timespec_wait
    printf("Waiting for 1.5 seconds...\n");
    timespec delay = timespec_from_ms(1500);
    timespec_wait(delay);
    printf("Done waiting!\n");

    return 0;
}