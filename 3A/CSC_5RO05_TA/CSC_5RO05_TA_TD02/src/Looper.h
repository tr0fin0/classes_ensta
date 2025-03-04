#ifndef LOOPER_H
#define LOOPER_H

#include "Timer.h"

class Looper : public Timer {
private:
    volatile bool doStop;   // Ensures the loop stops correctly
    volatile double iLoop;  // Prevents compiler optimizations removing updates
    double nLoops;          // Stores loop limit

protected:
    void callback() override;  // Timer callback function

public:
    Looper();
    double runLoop(double nLoops);
    void stopLoop();
    double getSample() const;  // Const method (does not modify state)
};

#endif // LOOPER_H
