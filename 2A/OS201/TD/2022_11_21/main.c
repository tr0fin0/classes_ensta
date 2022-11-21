#include <stdio.h>
#include "utils.h"
// question 9
#include <fcntl.h>
#include <unistd.h>


// question 1
#define STACK_SIZE 4096
char stack_0[STACK_SIZE];
char stack_1[STACK_SIZE];
char stack_2[STACK_SIZE];
char stack_3[STACK_SIZE];


int main(void) {
    printf("TD 21/11/2022\n");
    // question 9
    fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);

    // question 1
    printf("0x%p\n",(char *) &stack_0);
    printf("0x%p\n",(char *) &stack_1);
    printf("0x%p\n",(char *) &stack_2);
    printf("0x%p\n",(char *) &stack_3);


    // question 4 
    coroutine_t routine_0 = init_coroutine(*stack_0, STACK_SIZE, &fonction_0);
    coroutine_t routine_1 = init_coroutine(*stack_1, STACK_SIZE, &fonction_1);

    enter_coroutine(routine_0);
    enter_coroutine(routine_1);


    // question 9
    /*
    comment getchar() bloque le fonctionnement du système, les threads seront arretes quand il reçoit en entre de clavier. donc le processus s'arret
    */


    return 0;
};