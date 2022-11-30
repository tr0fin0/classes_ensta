# question 2
.global enter_coroutine
    enter_coroutine:
        mov %rcx, %rsp
        pop %r15
        pop %r14
        pop %r13
        pop %r12
        pop %rbx
        pop %rbp
        ret

# question 5
.global switch_coroutine
    switch_coroutine:
        push %rbp
        push %rbx
        push %r12
        push %r13
        push %r14
        push %r15
        mov %rsp,(%rcx)
        mov %rsi,%rcx
        jmp enter_coroutine
