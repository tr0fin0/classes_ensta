#include <pthread.h>
#include <iostream>


struct Data
{
    volatile bool stop;
    volatile double counter;
    pthread_mutex_t mutex;
};

void* incrementer(void* v_data)
{
    Data* p_data = (Data*) v_data;

    while (not p_data->stop)
    {
        pthread_mutex_lock(&p_data->mutex);
        p_data->counter += 1.0;
        pthread_mutex_unlock(&p_data->mutex);
    }

    return v_data;
}


int main()
{


    Data data = { false, 0.0 };
    pthread_mutex_init(&data.mutex, nullptr);

    pthread_t thread[3];
    pthread_create(&thread[0], nullptr, incrementer, &data);
    pthread_create(&thread[1], nullptr, incrementer, &data);
    pthread_create(&thread[2], nullptr, incrementer, &data);

    for(char cmd = 'r'; cmd != 's'; std::cin >> cmd)
    {
        std::cout << "type 's' to stop: " << std::flush;
    }

    data.stop = true;
    for(int i = 0; i < 3; ++i)
    {
        pthread_join(thread[i], nullptr);
    }
    pthread_mutex_destroy(&data.mutex);

    std::cout << "counter value: " << data.counter << std::endl;

}