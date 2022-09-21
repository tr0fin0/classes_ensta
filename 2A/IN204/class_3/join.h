#include <iostream>
using namespace std;


int joinDigits(int base, int newDigit)
{
    if ( !(0 <= newDigit && newDigit <= 9) )
        throw overflow_error("not a digit");

    return base*10 + newDigit%10;
};


template <typename T> T joinDigitsGeneric(T base, T newDigit)
{
    if ( !(0 <= newDigit && newDigit <= 9) )
        throw overflow_error("not a digit");

    return base*10 + newDigit;
};


template <> float joinDigitsGeneric(float base, float newDigit)
// float joinDigitsGeneric(float base, float newDigit)
{
    char buffer[20];
    int len = snprintf(buffer, sizeof(buffer), "%f", base);
    int lastDigitPos = 0;

    for (auto i = len - 1 ; i > 0 and buffer[i] == '0' ; --i)
        lastDigitPos = i;
    buffer[lastDigitPos] = newDigit + '0';
    buffer[lastDigitPos + 1] = '\0';

    printf(
        "%s: base = %f, newDigit = %f, buffer = %s\n",
        __PRETTY_FUNCTION__, base, newDigit, buffer
    );

    return atof(buffer);
};

template <int> int joinDigitsGeneric(int, int);
// template <float> float joinDigitsGeneric(float, float);