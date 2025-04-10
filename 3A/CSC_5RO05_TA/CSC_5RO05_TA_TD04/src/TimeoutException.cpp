#include "TimeoutException.h"

TimeoutException::TimeoutException(long timeout_ms)
    : std::runtime_error("Mutex lock timeout: " + std::to_string(timeout_ms) + " ms"), timeout_ms(timeout_ms) {}