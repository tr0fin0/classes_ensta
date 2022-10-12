template <typename T>
requires CONDITION 
void function(T PARAMETER) {
    ...
}

template <typename T>
void function(T PARAMETER) requires CONDITION {
    ...
}