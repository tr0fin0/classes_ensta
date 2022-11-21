#include <stdio.h>
#include "utils.h"


// question 1
#define STACK_SIZE 4096
char stack_0[STACK_SIZE];
char stack_1[STACK_SIZE];
char stack_2[STACK_SIZE];
char stack_3[STACK_SIZE];



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


    // 
    char stack[STACK_SIZE];


    return 0;
}