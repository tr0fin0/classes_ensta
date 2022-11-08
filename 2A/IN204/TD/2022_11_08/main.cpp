
#include <iostream>

#include "key_value_pair.hpp"


int main (void) {
    // question n1.1
    //  génère automatiquement
    //  constructeur vide responsable pour créer une instance sans valeurs pour les variables
    key_value_pair K0;

    // question n1.1
    //  pas génère automatiquement
    //  constructeur qui initialise l'objet avec des variables reçus dans le constructeur
    key_value_pair K1(5, "five");

    // question n1.1
    //  pas génère automatiquement
    //  constructeur qui initialise l'objet avec une autre objet de la même class
    key_value_pair K2(K1);


    // question n1.2

    // question n1.2

    // question n1.2


    return 0;
}