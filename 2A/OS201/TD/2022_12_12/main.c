#include <stdio.h>
#include <unistd.h>


char in_buffer[100];
char out_buffer[100];

int main(void) {

    // variables
    char *ptr = &in_buffer[0];
    int a, b, num;

    printf("main\n");


    while(1) {
        read(STDIN_FILENO, in_buffer, 1000);

        switch (*ptr++)
        {
        case '+':
            // printf("+\n");
            sscanf(ptr, "%d,%d", &a, &b);

            num = sprintf(out_buffer, "result: %d\n", a+b);

            write(STDOUT_FILENO, out_buffer, num);
            break;
        
        case '-':
            // printf("-\n");
            sscanf(ptr, "%d,%d", &a, &b);

            num = sprintf(out_buffer, "result: %d\n", a-b);

            write(STDOUT_FILENO, out_buffer, num);
            break;

        default:
            // printf("default\n");
            break;
        };
    };

    // int sscanf(const char *str, const char *format, ...);
    // int sprintf(char *str, const chart *format, ...);

    return 0;
};