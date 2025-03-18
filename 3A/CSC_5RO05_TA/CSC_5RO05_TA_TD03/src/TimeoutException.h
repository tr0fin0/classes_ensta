#ifndef TIMEOUTEXCEPTION_H
#define TIMEOUTEXCEPTION_H

#include <stdexcept>
#include <string>

class TimeoutException : public std::runtime_error
{
public:
    const long timeout_ms;

public:
    explicit TimeoutException(long timeout_ms);
};

#endif // TIMEOUTEXCEPTION_H
