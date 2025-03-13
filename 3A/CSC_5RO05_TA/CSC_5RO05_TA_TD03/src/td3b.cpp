#include "Mutex.h"

#include <iostream>
#include <thread>
#include <chrono>


void testMutex(Mutex& mutex)
{
    try
    {
        Mutex::Lock lock(mutex, 1000);

        std::cout << "Mutex locked successfully" << std::endl;
    } catch (const TimeoutException& e)
    {
        std::cerr << e.what() << std::endl;
    }
}


int main()
{
    Mutex mutex;
    std::thread t1(testMutex, std::ref(mutex));
    std::thread t2(testMutex, std::ref(mutex));

    t1.join();
    t2.join();

    return 0;
}