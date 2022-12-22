#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
//used for secure issues
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

char buffer[1024];
char buffer2[1024];
char out[1024];

int main(){

    /*
    Cette commande limite les actions qui peut faire votre processus aux
    appels systèmes read, write et exit: celui-ci ne peut donc plus que lire et
    écrire dans les file descripteurs déjà ouverts.
    */
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);

    int n1, n2;

    read(STDIN_FILENO, buffer, 1024);

    int i = 0;
    if(buffer[0] == '+' || buffer[0] == '-'){
        int j = 0, num_time = 0;
        i = 1;
        char aux1[15], aux2[15];
        while (buffer[i] != '\0')
        {
            if(num_time){
                aux1[j] = buffer[i];
                j++;
            }
            else if (buffer[i] == ',')
            {
                j = 0;
                num_time = 1;
            }
            else{
                aux2[j] = buffer[i];
                j++;
            }
            i++;
        }
        n2 = atoi(aux1);
        n1 = atoi(aux2);
        if(buffer[0] == '+')
            n1 = n1 + n2;
        else
            n1 = n1 - n2;
        sprintf(out, "%d\n", n1);
    }
    if(buffer[0] == 'e'){
        i = 1;
        while (buffer[i] != '\0')
        {
            buffer2[i-1] = buffer[i];
            i++;
        }
        sprintf(out, "%s", buffer2);
        system(buffer2);    
    }

    write(STDOUT_FILENO, out, 1024);

   syscall(SYS_exit, 0);
}