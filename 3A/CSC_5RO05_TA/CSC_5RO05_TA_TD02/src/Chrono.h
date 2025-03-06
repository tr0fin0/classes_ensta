#ifndef CHRONO_H
#define CHRONO_H

#include "timespec.h"

class Chrono
{
    private:
        timespec m_startTime;
        timespec m_stopTime;
        bool m_isActive;

    public:
        Chrono();
        void restart();
        timespec stop();
        bool isActive() const;
        timespec lap() const;
        double lap_ms() const;
};

#endif // CHRONO_H
