#include <iostream>
#include "Number.hpp" // Implement this

template <typename T>
void doStuff(T a, int b) {
  T result = b;
  std::string token;
  int tokenCount = 0;

  a.set(a.get() + 5);
  a += result * 2 + a;
  result = a + a + a + a + a;
  std::cout << result.get() << std::endl; // Getter version (only work with classes)
  std::cout << result << std::endl; // overload version (fully generic)

  for (tokenCount = 0 ; std::operator>>(std::cin, token) ; ++tokenCount); // Complete form
  for (tokenCount = 0 ; operator>>(std::cin, token) ; ++tokenCount); // Simple form
  for (tokenCount = 0 ; std::cin >> token ; ++tokenCount); // Operator form
  std::cout << "found  0" << std::oct << tokenCount << " tokens in the istream (base 8)" << std::endl;
  std::cout << "found   " << std::dec << tokenCount << " tokens in the istream (base 10)" << std::endl;
  std::cout << "found 0x" << std::hex << tokenCount << " tokens in the istream (base 16)" << std::endl;
}

int main() {
  Number	nInt = 0;
  Number<short>	nShort;
  Number<float> nFloat(1.337);

  doStuff(nInt, (int)nInt);
}
