#include<stdio.h>
#define STACK_SIZE 4096





char pile1[STACK_SIZE];
char pile2[STACK_SIZE];
char pile3[STACK_SIZE];
char pile4[STACK_SIZE];

typedef void * coroutine_t;


void enter_coroutine(coroutine_t coroutine);
void switch_coroutine(coroutine_t *ct1, coroutine_t ct2);
coroutine_t coroutine_ordo;
coroutine_t coroutine4;
coroutine_t coroutine2;
coroutine_t coroutine3;





struct thread{
    coroutine_t coroutine;
    int statut; //O prêt 1 bloqué

};


struct thread thread2;
struct thread thread3;
struct thread thread4;


//struct liste_thread{
  //  thread* premier;
//};

struct thread* thread_courant ;

//comme un push pour préparer le enter_coroutine
coroutine_t init_coroutine(void *stack_begin, unsigned int stack_size, void (*initial_pc)(void)){
char *stack_end = ((char *)stack_begin) + stack_size;
void* *ptr = stack_end;
ptr--;
*ptr = initial_pc; //program counter
ptr--;
//*ptr = 0; //rbp
ptr--;
//*ptr = 0; // rbx
ptr--;
//*ptr = 0; //r12
ptr--;
//*ptr = 0; //r13
ptr--;
//*ptr = 0; //r14
ptr--;
//*ptr = 0; //r15

return ptr;
}

void yield(void){
    switch_coroutine(&thread_courant->coroutine, coroutine_ordo);
}




void fonction_ordo(){
    while(1){
        thread_courant = &thread2;
        switch_coroutine(&coroutine_ordo, thread2.coroutine);
        thread_courant = &thread3;
        switch_coroutine(&coroutine_ordo, thread3.coroutine);
        thread_courant = &thread4;
        switch_coroutine(&coroutine_ordo, thread4.coroutine);
    }
    

}

void fonction2(){
    
    int compteur1 = 1;
    while(1){
        printf("je suis la coroutine1\n");
        printf("compteur = %d\n",compteur1);
        compteur1 = compteur1 +1;
        
        yield();//appel la coroutine2
    }
    
    

}

void fonction3(){
    int compteur2 = 1;
    
    while(1){

        printf("je suis la coroutine3\n");
        printf("compteur = %d\n",compteur2);
        ++compteur2;
        yield();//appel la coroutine1;
    }
    

}


void fonction4(){
   
    int compteur2 = 1;
    while(1){
        printf("je suis la coroutine4\n");
        printf("compteur = %d\n",compteur2);
        ++compteur2;
        yield();//appel la coroutine1;
    }
    

}

int main(){

    
    coroutine_ordo = init_coroutine(pile1, STACK_SIZE, &fonction_ordo); //ordonnanceur !
    printf("coroutine1 init\n");
    thread2.coroutine = init_coroutine(pile2, STACK_SIZE, &fonction2);
    printf("coroutine2 init\n");
    thread3.coroutine = init_coroutine(pile3, STACK_SIZE, &fonction3);
    printf("coroutine3 init\n");
    thread4.coroutine = init_coroutine(pile4, STACK_SIZE, &fonction4);
    printf("coroutine4 init\n");

    
    enter_coroutine(coroutine_ordo);
    
    return 0;
}
 
