#include <iostream>
using std::string;

class Animal
{
public:                     // visible for everyone / everything
    string Name;
    int Age;
    float Weight;

    Animal(string name, int age, float weight)
    {
        Name = name;
        Age = age;
        Weight = weight;
    };


    void print()
    {
        std::cout << "Name:   " << Name   << std::endl;
        std::cout << "Age:    " << Age    << std::endl;
        std::cout << "Weight: " << Weight << std::endl;
    };

protected:                  // visible by certain conditions


private:                    // only visible within the class / object


};