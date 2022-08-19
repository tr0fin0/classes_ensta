#include <stdio.h>



void pointers() {
    int x = 3;
    int y = 5;

    int *p = &y;
    int *q = &x;

    int **pp = &q;
    **pp = *q + **pp;   // x = 3 + 3
    *(&y) = *p + 1;     // y = 5 + 1


// \*arg[] est égale \**arg
};

void chars() {
    // pour finir une mot il le faut terminer avec '\0'
    // si on ne mets pas ça le programme va, probablement, retourner un "segment fault error"
    // encore pire, le programme va continuer et y retourner un valeur incorrecte sans erreur

    // il y a un longueur n+1 où n est la longueur du mot

    // un character c'est un octet
    // TODO chercher la taille de chaque type

    char *s = malloc(5*sizeof(char));    // "test" -> 't' 'e' 's' 't' '\0'

};



void structs() {
    // s -> c   avec la notation plus commune
    // (*u).c   avec un pointer CORRECTE
    // TODO chercher si struct est un objet

    struct t {
        int a;
        int b;
    };
};

void static_allocation() {
    // int  t(10) pas valide en C
    // int  t{10} pas valide en C
    // int *t[10] tableau de pointers de int's
    // int  t[10] tableau de int's CORRECTE
};



void acess() {
    // t[i]
    // \*(t+i)
    // \* t+i   change le valeur dedans 
    // \*((t)+i)
};


void prints() {
    char *argv[4] = "test";
    printf("%s\n", argv[0]);

    // TODO reviser les options de printf
    // TODO reviser pop et push
};


void variables() {
    // int
    // short
    // long
    // unsigned int
    // unsigned short
    // unsigned long

    // il n'aura pas de mensage d'erreur pendant l'execution quand quelques variables sont couples pendant l’attribution des variables

    // unsigned to signed will always loss information because the interpretation of the result is different among representations
    // unsigned long -> long
    // int -> unsigned int
    // int -> short
    // short -> int CORRECTE
};


void operations() {
    // && and logic
    // & and  bit a bit
};


void arbre_binary() {
    struct abre {
        int value;
        struct abre *gauche;
        struct abre *droite;
    };
};



int main(void) {
    pointers();
    chars();
    structs();
    static_allocation();

    // TODO search estrutures des donnes
    // TODO search hashtables
    

    return 0;   // le programme doit retourner 0 s'il n'avait pas des problèmes pendant son execution
};



main();