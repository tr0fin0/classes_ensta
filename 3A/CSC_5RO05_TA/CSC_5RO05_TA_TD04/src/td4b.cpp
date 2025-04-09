#include "Semaphore.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>


class TokenProducer
{
private:
    Semaphore& semaphore;
    const int tokensToProduce;
    int producedCount;

public:
    TokenProducer(Semaphore& s, int tokens) :
        semaphore(s),
        tokensToProduce(tokens),
        producedCount(0)
    {}

    void operator()()
    {
        for (int i = 0; i < tokensToProduce; ++i)
        {
            semaphore.give();
            producedCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    int getProducedCount() const { return producedCount; }
};

class TokenConsumer
{
private:
    Semaphore& semaphore;
    int consumedCount;

public:
    TokenConsumer(Semaphore& s) : semaphore(s), consumedCount(0) {}

    void operator()()
    {
        while (true)
        {
            // TODO timed take
            if (semaphore.take(500))
            {
                consumedCount++;
            } else {
                break;
            }
        }
    }

    int getConsumedCount() const { return consumedCount; }
};


int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <num_consumers> <num_producers> <tokens_per_producer>\n";
        return 1;
    }

    const int numConsumers = std::stoi(argv[1]);
    const int numProducers = std::stoi(argv[2]);
    const int tokensPerProducer = std::stoi(argv[3]);

    // TODO remove negative check
    Mutex mutex;
    Monitor monitor(mutex);
    Semaphore semaphore(monitor, mutex, 0, std::numeric_limits<int>::max());

    std::vector<TokenProducer> producers;
    std::vector<TokenConsumer> consumers;
    std::vector<std::thread> threads;

    for (int i = 0; i < numProducers; ++i)
        producers.emplace_back(semaphore, tokensPerProducer);

    for (int i = 0; i < numConsumers; ++i)
        consumers.emplace_back(semaphore);

    for (auto& consumer : consumers)
        threads.emplace_back(std::ref(consumer));

    for (auto& producer : producers)
        threads.emplace_back(std::ref(producer));

    for (auto& t : threads)
        t.join();


    int totalProduced = 0;
    for (auto& p : producers)
    {
        totalProduced += p.getProducedCount();
        std::cout << "Producer created " << p.getProducedCount() << " tokens\n";
    }

    int totalConsumed = 0;
    for (auto& c : consumers)
    {
        totalConsumed += c.getConsumedCount();
        std::cout << "Consumer took " << c.getConsumedCount() << " tokens\n";
    }


    std::cout << "\nTotal tokens produced: " << totalProduced << "\n";
    std::cout << "Total tokens consumed: " << totalConsumed << "\n";
    std::cout << "Balance: " << (totalProduced - totalConsumed) << "\n";

    if (totalProduced == totalConsumed)
    {
        std::cout << "SUCCESS: No tokens were lost!\n";
    } else
    {
        std::cout << "ERROR: Token count mismatch!\n";
    }

    return 0;
}
