#include <iostream>

template<typename K = int, typename V = std::string>  // default template
class keyValuePair {
    private:
        K key;
        V value;

    public:
        keyValuePair();
        keyValuePair(K theKey, V theValue);
        keyValuePair(const keyValuePair& anotherPair);


        // getters 
        K getKey(void){
            return (*this).key;
        };
        
        V getValue(void){
            return (*this).value;
        };


        // setters
        void setKey(K const & newKey){  // use const to be sure that the variable is constant
            (*this).key = newKey;
        };

        void setValue(V const & newValue){
            (*this).value = newValue;
        };


        // operators
        bool operator == (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey == rKey);
        }

        bool operator != (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey != rKey);
        }

        bool operator <  (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey < rKey);
        }

        bool operator <= (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey <= rKey);
        }

        bool operator >  (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey > rKey);
        }

        bool operator >= (const keyValuePair& rightPair)
        {
            K lKey = (*this).key;
            K rKey = (rightPair).key;

            return (lKey >= rKey);
        }
};