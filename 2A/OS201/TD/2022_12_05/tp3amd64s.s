.globl enter_coroutine, switch_coroutine

/* Note: le premier argument est dans ecx, le deuxieme dans edx. */
switch_coroutine:
        push %rbp
        push %rbx
        push %r12
        push %r13
        push %r14
        push %r15               
        mov %rsp,(%rdi)         /* Store stack pointer to the coroutine pointer.. */
        mov %rsi,%rdi           /* Continue to enter_coroutine, mais echange les arguments d'abord. */
        
enter_coroutine:
        mov %rdi,%rsp         /* Load the stack pointer from the coroutine pointer. */
        pop %r15
        pop %r14
        pop %r13
        pop %r12
        pop %rbx
        pop %rbp
        ret                   /* Pop the program counter. */


        
