#ifndef COUNTDOWN_H
#define COUNTDOWN_H

#include "Timer.h"

class CountDown : public Timer
{
// ? should be private
public:
    int count;

protected:
    void callback() override;

public:
    CountDown(int n);
};

#endif // COUNTDOWN_H
