typedef void * coroutine_t;


// question 2
void enter_coroutine(coroutine_t cr);
/* 
    Quitte le contexte courant et charge les registres et la pile de CR.
*/

// question 5
void switch_coroutine(coroutine_t *p_from, coroutine_t to);
/*
    Sauvegarde le contexte courant dans p_from, et entre dans TO.
*/



// question 4
void fonction_0(void) {
    int i = 0;
    while(1) {
        printf("0: %d\n", i++);
    };
};

// question 6
void fonction_1(void) {
    int j = 0;
    while(1) {
        printf("1: %d\n", j++);
    };
};


// question 3
coroutine_t init_coroutine(
        void *stack_begin, 
        unsigned int stack_size, 
        void (*program_counter)(void)
    ) {
    /*
        Initialise la pile et renvoie une coroutine telle que, lorsqu’on entrera dedans, elle commencera à s’exécuter à l’adresse program_counter.
    */
    char *stack_end = ((char *)stack_begin) + stack_size;
    void **ptr = stack_end;

    ptr--;
    *ptr = program_counter;
    
    // loop used to avoid code repetition 
    for (int i = 0; i < 6; i++) {
        ptr--;
        *ptr = 0;   // arbitrary value NULL
    };

    return ptr;
};


// question 7
static __thread varThread;

// question 8
void yield(void) {

};

// question 8
void thread_create(void) {

};