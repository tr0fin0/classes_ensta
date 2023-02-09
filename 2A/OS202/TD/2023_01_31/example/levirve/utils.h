#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#ifdef _DEBUG
#define DEBUG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG(str) do { } while ( false )
#endif
#ifndef INFO
#define INFO(str) do { std::cout << str << std::endl; } while( false )
#else
#define INFO(str) do { } while ( false )
#endif

struct ComplexNum {
    double real, imag;
};

extern int num_thread, width, height;
extern double real_min, imag_min, dx, dy;
extern bool gui;

void initial_env(int argc, char** argv);

class Timer {
private:
	time_point<system_clock> s;
public:
    void start();
    unsigned int stop();
    void log();
};

#endif