# question 2
.global enter_coroutine
    enter_coroutine:
        mov %rdi, %rsp
        pop %r15
        pop %r14
        pop %r13
        pop %r12
        pop %rbx
        pop %rbp
    ret

