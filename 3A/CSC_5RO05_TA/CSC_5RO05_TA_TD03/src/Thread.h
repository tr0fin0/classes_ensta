#ifndef THREAD_H
#define THREAD_H

#include <pthread.h>


class Thread
{
public:
    const int id;

private:
    bool started;
    pthread_t posixThreadId;
    pthread_attr_t posixThreadAttrId;


public:
    Thread(int id);
    virtual ~Thread();
    void start(int priority = 0);
    void join();
    bool isStarted();
    long duration_ms();

protected:
    virtual void run();

private:
    static void* call_run(void* v_Thread);
    static int getMaxPrio(int policy);
    static int getMinPrio(int policy);
    static void setMainSched(int policy);
    static int getMainSched(int policy);
};

#endif // THREAD_H
