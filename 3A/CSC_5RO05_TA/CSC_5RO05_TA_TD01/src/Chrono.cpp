#include "Chrono.h"

Chrono::Chrono()
{
    restart();
}

void Chrono::restart()
{
    clock_gettime(CLOCK_MONOTONIC, &m_startTime);
    m_stopTime = m_startTime;
    m_isActive = true;
}

timespec Chrono::stop()
{
    if (m_isActive)
    {
        clock_gettime(CLOCK_MONOTONIC, &m_stopTime);
        m_isActive = false;
    }
    return lap();
}

bool Chrono::isActive() const
{
    return m_isActive;
}

timespec Chrono::lap() const
{
    timespec now;
    if (m_isActive)
    {
        clock_gettime(CLOCK_MONOTONIC, &now);
    }
    else
    {
        now = m_stopTime;
    }

    timespec result = timespec_normalize(now - m_startTime);

    return result;
}

double Chrono::lap_ms() const
{
    timespec elapsed = lap();

    return elapsed.tv_sec * 1000.0 + elapsed.tv_nsec / 1000000.0;
}
