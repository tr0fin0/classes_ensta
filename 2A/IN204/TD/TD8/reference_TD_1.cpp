#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
std::mutex g_mutex;



void sleep_ms(size_t ms) 
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
void wait_for_order(int i, bool const & do_action, void (*action)(int)) 
{
    std::stringstream logMsg;
    std::cout   << " COUT: Thread " << i << " -" // Multiple operator<<() called
                << " is running with id " << std::this_thread::get_id() << std::endl;

    logMsg  << "SSTREAM: Thread " << i << " -" // operator<<() called once
            << " is running with id " << std::this_thread::get_id() << std::endl;
    // "Atomic" print
    std::cout << logMsg.str();
    // "spinLock"

    while (!do_action) 
    {

    }

    // All at once
    action(i);
    sleep_ms(500);
    // One at a time

    { // stack scope for the mutex guard
        std::lock_guard<std::mutex> lock(g_mutex);
        action(i*-1);
    }
}


int main() 
{
    int numThreads = 5;
    bool do_action = false;
    std::vector<std::thread> threads;

    for (int i = 0 ; i < numThreads ; ++i) 
    {
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        // https://isocpp.org/wiki/faq/cpp11-language#lambda
        // https://en.cppreference.com/w/cpp/utility/functional/ref
        threads.push_back(std::thread(wait_for_order, i, std::cref(do_action),
        [](int i){std::cout << "Hello from " << i << std::endl;; }));
    }

    sleep_ms(100); // Just to be sure all threads have printed, for the demo
    std::cout << "Main thread with id " << std::this_thread::get_id() << " is sleeping" << std::endl;
    sleep_ms(1000);
    std::cout << "[ ] Main thread is releasing the threads: (spammy action)" << std::endl << std::endl;
    do_action = true; // Trigger the threads
    sleep_ms(250);

    std::cout << std::endl << "[ ] Async call of action should be done by now (clean)" << std::endl << std::endl;

    while (threads.size()) 
    {
        threads.back().join(); // main process thread waits for the thread to finish
        threads.pop_back();
    }
}