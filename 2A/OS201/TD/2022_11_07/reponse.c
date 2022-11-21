
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>


struct fs_header {
  char magic[8];
  uint32_t size;
  uint32_t checksum;
  char name[];
} __attribute__((packed));

struct file_header {
  uint32_t next;
  uint32_t spec_info;
  uint32_t size;
  uint32_t checksum;
  char name[];
} __attribute__ ((packed));

// an enum for possible file types
enum file_type {
  ft_hard_link=0,
  ft_directory,
  ft_regular,
  ft_symlink,
  ft_blockdev,
  ft_chardev,
  ft_socket,
  ft_fifo,
};

// return the file type of the corrsponding header
enum file_type ft(struct file_header* file) {
  // low 3 bits of the last byte in big endian, so low three bits of the first
  // byte in native endianness
  return (file->next>>24) & 7;
}

// converts an integer from fs endianness (big endian) to native endianness
// (little endian)
uint32_t fs_to_native(uint32_t x) {
  uint8_t *cast = (uint8_t*)&x;
  return (cast[0]<<24) | (cast[1]<<16) | (cast[2]<<8) | cast[3];
}

// aligns the address to the next 16-bytes boundary
long ceil_16(long x) {
  if (x&15L) {
    return (x|15L) + 1L;
  } else {
    return x;
  }
}

// returns the file header at the specified offset. Cleans the permission bits
// stored in the 3 least significant bits. Returns NULL if this is the last file.
struct file_header *file_from_offset(struct fs_header* fs, uint32_t offset) {
  uint32_t real_offset = fs_to_native(offset) & ~15;
  // Checks that the offset is sensible, instead of blindly segfaulting
  assert(real_offset < fs->size);
  if (real_offset) {
    return (struct file_header*)((char*)fs + real_offset);
  } else {
    return NULL;
  };
}

// returns a pointer to the start of the data of the file
char *file_data(struct file_header *file) {
  return (char*)ceil_16((long)&file->name[strlen(file->name)+1]);
}

// returns the file after the argument, or NULL if this is the last one.
struct file_header *next(struct fs_header *fs, struct file_header *file) {
  return file_from_offset(fs, file->next);
}

// pretty prints a file to stdout (name, type, size, but not content).
void pp_file(struct file_header* file) {
  char *type;
  switch (ft(file)) {
    case ft_hard_link: type = "hard link"; break;;
    case ft_directory: type = "directory"; break;;
    case ft_regular:   type = "regular"; break;;
    case ft_symlink:   type = "symlink"; break;;
    case ft_blockdev:  type = "block device"; break;;
    case ft_chardev:   type = "char device"; break;;
    case ft_socket:    type = "socket"; break;;
    case ft_fifo:      type = "fifo"; break;;
    default:           type = "impossible";
    }
  printf("%s (%s, %d bytes)\n", file->name, type, fs_to_native(file->size));
}

// lists the files in a directory on stdout.
void ls(struct fs_header *fs, struct file_header* dir) {
  assert(ft(dir)==ft_directory);
  printf("Listing content of directory %s\n", dir->name);
  struct file_header *file = file_from_offset(fs, dir->spec_info);
  while (file) {
    pp_file(file);
    file = next(fs, file);
  }
}

// find the first file with the specified name. Does not follow links. Returns
// NULL if no file matches.
struct file_header *find(struct fs_header *fs, struct file_header* root, char *name) {
  assert(ft(root)==ft_directory);
  struct file_header *file = file_from_offset(fs, root->spec_info);
  while (file) {
    // pas besoin de vérifier .. parce que .. est un hard_link
    if (ft(file)==ft_directory && strcmp(file->name, ".")) {
      struct file_header *ret = find(fs, file, name);
      if (ret) return ret;
    }
    if (ft(file)==ft_regular && !strcmp(file->name, name)) {
      return file;
    }
    file = next(fs, file);
  }
  return NULL;
}

void decode(struct fs_header *fs, size_t size) {
  // check that this file is really a romfs
  assert(strncmp(fs->magic, "-rom1fs-", sizeof(fs->magic))==0);
  // check that the claimed size makes sense.
  uint32_t inner_size = fs_to_native(fs->size);
  assert(inner_size <= size);
  int name_len = strlen(fs->name);
  // fs is aligned because mmap guarantees it, so aligning from the start of
  // the file is the same as aligning the pointer.
  assert(!((long)fs & 4095L));
  struct file_header *root = (struct file_header*)ceil_16((long)&fs->name[name_len+1]);

  printf("Root directory: ");
  pp_file(root);
  printf("------\n");
  ls(fs, root);

  printf("Looking for message.txt\n");
  struct file_header *message = find(fs, root, "message.txt");
  assert(message);
  printf("Found it!\n");
  pp_file(message);
  int file_size = fs_to_native(message->size);
  char* data = file_data(message);
  fwrite(data, 1, file_size, stdout);
}



int main(void){

  int fd = open("fs.romfs",O_RDONLY);
  assert(fd != -1);
  off_t fsize;
  fsize = lseek(fd,0,SEEK_END);

  //  printf("size is %d", fsize);
  void *addr = mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
  assert(addr != MAP_FAILED);
  decode(addr, fsize);
  return 0;
}
