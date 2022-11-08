#include <iostream>

class keyValuePair {
    private:
        int key;
        std::string value;

    public:
        keyValuePair();
        keyValuePair(int theKey, std::string theValue);
        keyValuePair(const keyValuePair& anotherPair);

        // getters 
        int getKey(void){
            return (*this).key;
        };
        
        std::string getValue(void){
            return (*this).value;
        };

        // setters
        void setKey(int newKey){
            (*this).key = newKey;
        };

        void setValue(std::string newValue){
            (*this).value = newValue;
        };
};