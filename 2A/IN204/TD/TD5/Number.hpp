#include <cstdio>
#include <iostream>


template<typename T>    // default template value
class Number
{
    private:
        T N;

    public:

        // default constructor
        Number() : N(0) {}
        
        // constructor with int
        Number(T const &N) : N(N) {}

        // constructor by copy
        Number(const Number &number)
        {
            this->N = number.N;
        }

        ~Number()
        {
            // destructor
        }



        void print()
        {
            std::cout << "N: " << this->N << std::endl;
        };

        T get(void) const           // implicitly receives this as input variable
        {
            return (this->N);
        };

        void set(T const & newVal)  // use const to be sure that the variable is constant
        {
            this->N = newVal;
        };



        Number operator + (const Number& rightValue)
        {
            Number result = *this;
            result.N = N + rightValue.N;

            return result;
        }

        Number operator - (const Number& rightValue)
        {
            Number result = *this;
            result.N = N - rightValue.N;

            return result;
        }

        Number operator * (const Number& rightValue)
        {
            Number result = *this;
            result.N = N * rightValue.N;

            return result;
        }

        Number operator / (const Number& rightValue)
        {
            Number result = *this;
            result.N = N / rightValue.N;

            return result;
        }

        // method friend can access private variables inside the class
        template <typename U>
        friend std::ostream& operator<<(std::ostream& stream, const Number<U>& number);
};

template <typename T>
std::ostream& operator<<(std::ostream& output, const Number<T>& number)
{
    return output << number.N;
};