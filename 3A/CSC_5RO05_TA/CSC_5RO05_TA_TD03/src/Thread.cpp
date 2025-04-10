#include "Thread.h"

Thread::Thread(int id) : id(id), started(false)
{
    pthread_attr_init(&posixThreadAttrId);
}

Thread::~Thread()
{
    pthread_attr_destroy(&posixThreadAttrId);
}

void* Thread::call_run(void* p_thread)
{
    Thread* thread = static_cast<Thread*>(p_thread);
    thread->run();

    return nullptr;
}

void Thread::start(int priority)
{
    if (not started)
    {
        started = true;
        pthread_create(&posixThreadId, &posixThreadAttrId, Thread::call_run, this);
    }
}

void Thread::join()
{
    if (not started)
        pthread_join(posixThreadId, nullptr);
}

bool Thread::isStarted()
{
    return started;
}

long Thread::duration_ms()
{

    return 0;
}

int Thread::getMaxPrio(int policy)
{
    return sched_get_priority_max(policy);
}

int Thread::getMinPrio(int policy)
{
    return sched_get_priority_min(policy);
}

void Thread::setMainSched(int policy)
{
    struct sched_param param;

    param.sched_priority = getMaxPrio(policy);
    sched_setscheduler(0, policy, &param);
}
