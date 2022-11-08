class key_value_pair {
    private:
        int key;
        std::string value;

    public:
        key_value_pair();
        key_value_pair(int theKey, std::string theValue);
        key_value_pair(const key_value_pair& anotherPair);
};