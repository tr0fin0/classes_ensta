#ifndef LOOPER_H
#define LOOPER_H

class Looper
{
private:
    volatile bool doStop;
    volatile double iLoop;

public:
    Looper();
    double runLoop(double nLoops);
    void stopLoop();
    double getSample() const;
};

#endif // LOOPER_H
