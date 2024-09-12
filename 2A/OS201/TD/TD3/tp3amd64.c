
#include <stdlib.h>

typedef void * coroutine_t;

#define STACK_SIZE 4096

char stack1[STACK_SIZE] __attribute__((aligned(4096)));
char stack2[STACK_SIZE] __attribute__((aligned(4096)));
char stack3[STACK_SIZE] __attribute__((aligned(4096)));
char stack4[STACK_SIZE] __attribute__((aligned(4096)));

coroutine_t co1 = &stack1[STACK_SIZE];
coroutine_t co2 = &stack2[STACK_SIZE];
coroutine_t co3 = &stack3[STACK_SIZE];
coroutine_t co4 = &stack4[STACK_SIZE];

void thread_exit_function(void){
  printf("Error: Exited from a co-routine\n");
  exit(1);
}

/* Initialize. Specific to X86 32 bit.  */
void init_coroutine(coroutine_t *p_cr, void (*initial_pc)(void)){
  void *initial_sp = *p_cr;
  void **cr = *p_cr;
  cr --;
  *cr = thread_exit_function;   /* Just in case. */
  cr --;
  *cr = thread_exit_function;   /* Just in case. */
  cr --;
  *cr = initial_pc;             /* PC */
  cr --;
  *cr = initial_sp;             /* RBP */
  cr --;                        /* RBX */
  *cr = 0x22222222;
  cr --;                        /* R12 */
  *cr = 0x33333333;
  cr --;                        /* R13 */
  *cr = 0x44444444;
  cr --;                        /* R14 */
  *cr = 0x55555555;
  cr --;                        /* R15 */
  *cr = 0x66666666;
  *p_cr = cr;
}
void /* __attribute__((fastcall)) */ enter_coroutine(coroutine_t cr);
void /* __attribute__((fastcall)) */ switch_coroutine(coroutine_t *from, coroutine_t to);


#if 1 //SECTION_1

/* Test co-routines. */


void a_function(void){
  int i = 0;
  while(1) {
    i ++;
    printf("In A %d \n", i);
    printf("Co1 co2 %p %p\n", co1, co2);
    switch_coroutine(&co1,co2);
  }
}

void b_function(void){
  int i = 0;
  while(1)  {
    i ++;
    printf("In B %d \n", i);
    switch_coroutine(&co2,co1);
  }
}

void main(void){
  init_coroutine(&co1, a_function);
  printf(" Co1 %p %p Co2 %p %p\n", &co1, co1, &co2,co2);  
  init_coroutine(&co2, b_function);
  printf(" Co1 %p %p Co2 %p %p\n", &co1, co1, &co2,co2);
  enter_coroutine(co2);
}
#endif

#if 0 // SECTION2

/* NB: Une fois les co-routines faites, j'ai mis 20 minutes a faire la partie thread. */

enum status { 
  STATUS_READY,
  STATUS_BLOCKED,
};


struct thread {
  coroutine_t cr;
  enum status status;
};

/* The scheduler */
coroutine_t scheduler = &stack1[STACK_SIZE];

/* The normal tasks. */
struct thread thread2 = { .cr = &stack2[STACK_SIZE], .status = STATUS_READY };
struct thread thread3 = { .cr = &stack3[STACK_SIZE], .status = STATUS_READY };
struct thread thread4 = { .cr = &stack4[STACK_SIZE], .status = STATUS_READY };


struct thread *current_thread;

void yield(void){
  /* printf("yield\n"); */
  switch_coroutine(&current_thread->cr,scheduler);
}


void scheduler_rtn(void){

  struct thread * threads[] = {&thread2,&thread3,&thread4};

  int current = 0;
  while(1){
    struct thread *fb = threads[current];
    switch(fb->status){
    case STATUS_READY:
      /* printf("Entering task %d\n", current); */
      current_thread = fb;
      switch_coroutine(&scheduler, (fb->cr));
      break;

    default:
      /* printf("Skipping task %d\n", current); */
      break;
    }
    current = (current + 1) % 3;
  }
}

void init_thread(struct thread *th, void (*initial_pc)(void)){
  init_coroutine(&th->cr,initial_pc);
  th->status = STATUS_READY;
}


#if 0// SECTION_2

void taska(void){
  int i = 0;
  while(1){
    printf("In task a: %d\n", i);
    yield();
    i++;
  }
}

void taskb(void){
  int i = 0;
  while(1){
    printf("In task b: %d\n", i);
    yield();
    i++;
  }
}

void taskc(void){
  int i = 0;
  while(1){
    printf("In task c: %d\n", i);
    yield();
    i++;
  }
}

#include <unistd.h>
#include <fcntl.h>

int main(void){

  fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
  
  init_coroutine(&scheduler,scheduler_rtn);  
  init_thread(&(thread2),taska);
  init_thread(&(thread3),taskb);
  init_thread(&(thread4),taskc);    
  enter_coroutine(scheduler);

  return 0;
}
#endif

#endif


#if 0

#ifdef NAIVE

struct semaphore 
{
  volatile int counter;
};


void sem_wait(struct semaphore *sem)
{
  while(1){
    int old = sem->counter;
    if(old != 0) {
      sem->counter = old - 1;
      break;
    }
    yield();
  }
}

void sem_signal(struct semaphore *sem)
{
  sem->counter = sem->counter + 1;
}

#else  /* OPTIMIZED */

struct semaphore 
{
  volatile int counter;
  volatile struct thread *waiting;       /* NULL if no one is waiting. */
};

void sem_wait(struct semaphore *sem)
{
  while(1){
    int old = sem->counter;
    if(old != 0) {
      sem->counter = old - 1;
      break;
    }
    else {
      if(sem->waiting == 0){
        sem->waiting = current_thread;
        current_thread->status = STATUS_BLOCKED;
      }
    }
    yield();
  }
}

void sem_signal(struct semaphore *sem)
{
  sem->counter = sem->counter + 1;
  if(sem->waiting){
    sem->waiting->status = STATUS_READY;
    yield();                    /* optional. */
  }
}

#endif

struct queue {
  struct semaphore free;
  struct semaphore filled;
  char buffer[1];
};


/* Simple producer-consumer buffer with a single cell */
char input;

/* Count the free cells or the used cells. */
struct semaphore sem_input_free = {.counter=1,.waiting=0}, sem_input_filled={.counter=0,.waiting=0};

void put_input(char c){
  sem_wait(&sem_input_free);
  input = c;
  sem_signal(&sem_input_filled);
}

char get_input(void){
  sem_wait(&sem_input_filled);
  char res = input;
  sem_signal(&sem_input_free);
  return res;
}

char *output;

struct semaphore sem_output_free = {.counter=1,.waiting=0}, sem_output_filled={.counter=0,.waiting=0};

void put_output(char * c){
  sem_wait(&sem_output_free);
  output = c;
  sem_signal(&sem_output_filled);
}

char * get_output(void){
  sem_wait(&sem_output_filled);
  char *res = output;
  sem_signal(&sem_output_free);
  return res;
}



void input_task2(void){  
  while(1) {
    char c = getchar();
    put_input(c);
  }
}
void input_task(void){
  input_task2();
}


void output_task2(void){
  while(1) {
    char *s = get_output();
    while(*s){
      putchar(*s);
      /* Note: ne marche pas pour printf; peut-etre qu'il utilise des registres en plus. */
      /* printf("==%c==\n",*s); */
      s++;
      yield();                 /* Optional, is in the semaphores. */
    }
  }
}
void output_task(void){
  output_task2();
}


char buffer[8];

void main_task2(void){
  while(1){
    char c = get_input();
    buffer[0] = c; buffer[1] = c; buffer[2] = c; buffer[3] = 0;
    buffer[4] = c; buffer[5] = c; buffer[6] = c; buffer[7] = 0;    
    put_output(strdup(buffer));
    yield();
  }
}
void main_task(void){
  main_task2();
}


int main(void){
  init_coroutine(&scheduler,scheduler_rtn);  
  init_thread(&(thread2),input_task);
  init_thread(&(thread3),output_task);
  init_thread(&(thread4),main_task);    
  enter_coroutine(scheduler);

  return 0;
}

#endif
