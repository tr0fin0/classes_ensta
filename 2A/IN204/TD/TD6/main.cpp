#include <iostream>

#pragma once
#include<algorithm>
 
// template<typename  iterator>
// void simple_sort(iterator start, iterator end) 
//     requires(std::forward_iterator<iterator> && std::input_or_output_iterator<iterator>)
// {
//     for(;start != end; start ++)
//     {
//         auto it = start; it++;
//         for(;it != end; it ++)
//         {
//             // Compare si les deux elements sont dans le bon ordre.
//             if (*start > *it)
//                 std::swap(*start, *it);
//         };
//     };
// };


template<typename T>
concept Printable = requires(std::ostream& os, const T& msg)
{
    {os << msg};
};template <Printable T>
void print(const T& msg){
    std::cout << msg;
}template<Printable T>
void foo(T obj){ /* do something *}

int main()
{
    std::cout << "TD 12/10/2022" << std::endl << std::endl;

    


    
    return 0;
};