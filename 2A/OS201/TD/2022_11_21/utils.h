typedef void * coroutine_t;


// question 2
void enter_coroutine(coroutine_t cr);
/* 
    Quitte le contexte courant et charge les registres et la pile de CR.
*/
    .global enter_coroutine // visible to the linker
    enter_coroutine:
    mov %rdi, %rsp  /* RDI contains the argument to enter_coroutine. */
                    /* And is copied to RSP. */
    pop %r??
    pop %r??
    pop %r??
    pop %r??
    pop %r??
    pop %r??
    ret             /* Pop the program counter */
};


void switch_coroutine(coroutine_t *p_from, coroutine_t to) {
/*
    Sauvegarde le contexte courant dans p_from, et entre dans TO.
*/

    .global switch_coroutine // Makes switch_coroutine visible to the linker
    switch_coroutine:
    push %r??
    push %r??
    push %r??
    push %r??
    push %r??
    push %r??
    mov %rsp,(%rdi) /* Store the stack pointer to *(first argument) */
    mov %rsi,%rdi
    jmp enter_coroutine /* Call enter_coroutine with the second argument. */
};


coroutine_t init_coroutine(
        void *stack_begin, 
        unsigned int stack_size, 
        void (*initial_pc)(void)
    ) {
/*
    Initialise la pile et renvoie une coroutine telle que, lorsqu’on entrera dedans, elle commencera à s’exécuter à l’adresse initial_pc.
*/
    char *stack_end = ((char *)stack_begin) + stack_size;
    void **ptr = stack_end;

    ptr--;
    *ptr = initial_pc;

    ptr--;
    // *ptr = ...

    ptr--;
    // ...
    return ptr;
};