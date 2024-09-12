#include <cstdio>
#include <iostream>


template<typename T = int>  // default template value
class Number
{
    private:
        T C;

    public:
        typedef T value_type;

        Number() : C(0) {}    // default constructor
        Number(T const &C) : C(C) {}
        Number(const Number &number)
        {
            this->C = number.C;
        }

        void print()
        {
            std::cout << "C: " << this->C << std::endl;
        };

        T get(void) const   // implicitly receives this as input variable
        {
            return (this->C);
        };

        void set(T const & newVal)  // use const to be sure that the variable is constant
        {
            this->C = newVal;
        };
};