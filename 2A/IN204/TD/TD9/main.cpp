
#include <iostream>

#include "keyValuePair.hpp"


int main (void) {
    std::cout << "TD 08/11/2022" << std::endl << std::endl;

    /*
    Q 1.
        1
        génère automatiquement
        constructeur vide responsable pour créer une instance sans valeurs pour les variables
    */
    keyValuePair K0();

    /*
        1
        pas génère automatiquement
        constructeur qui initialise l'objet avec des variables reçus dans le constructeur
    */
    keyValuePair K1(5, "five");

    /*
        1
        pas génère automatiquement
        constructeur qui initialise l'objet avec une autre objet de la même class
    */
    keyValuePair K2(K1);


    /*
        2
    
    */

    /*
    Q 2.
        1
        non, ce n'est pas possible parce que ces variables sont definis sour le modificateur de visibilité private qui n'est pas accessible que pour l'objet lui même
        "private can be accessed only by the member functions inside the class"
        génère automatiquement
        "protected access modifier is similar to the private access modifier in the sense that it can’t be accessed outside of its class unless with the help of a friend class. The difference is that the class members declared as Protected can be accessed by any subclass (derived class) of that class as well"
        constructeur vide responsable pour créer une instance sans valeurs pour les variables
        "All the class members declared under the public specifier will be available to everyone"

        2
        getters and setters defined in the class
    */

    std::cout << K1.getKey();
    std::cout << K1.getValue();



    return 0;
}