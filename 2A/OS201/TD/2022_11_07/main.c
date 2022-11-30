#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>   // package only for Unix
#include <stdint.h>


void decode(struct fs_header *p, size_t size){

}


int main(void){
    int fd = open("fs.romfs",O_RDONLY);
    assert(fd != -1);
    off_t fsize;
    fsize = lseek(fd,0,SEEK_END);

    printf("size is %d", fsize);
    
    char *addr = mmap(addr, fsize, PROT_READ, MAP_SHARED, fd, 0);
    assert(addr != MAP_FAILED);
    decode(addr, fsize);

    return 0;
}