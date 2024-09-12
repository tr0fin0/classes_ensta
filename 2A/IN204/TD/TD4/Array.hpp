#include <cstdio>
#include <iostream>

typedef size_t size_type;


template <typename T, size_type S>
class Array
{
    private:
        T _buffer[S];

    public:
        void fill( const T & value )
        {
            for (size_type i = 0 ; i < S ; ++i)
            {
                _buffer[i] = value;
            };
        };

        void print()
        {
            std::string output = "[ ";

            for (size_type i = 0 ; i < S ; ++i)
            {
                output += std::to_string(_buffer[i]) + " ";
            };

            std::cout << output + "]" << std::endl;
        }

        T & operator[](size_type pos)
        {
            return _buffer[pos];
        };

        T const & operator[](size_type pos) const
        {
            return _buffer[pos];
        };

        size_type size() const
        {
            return S;
        }
};