#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Cell of a linked-list of words. */
struct list_cell_t {
  char *word ;                /* The word. */
  struct list_cell_t *next ;  /* Pointer to the next cell or NULL if none. */
};


/** Add the word 'word' in head of the list 'lst'. The word is copied in the
    cell. This function do allocate memory for the word. Hence 'word' is not
    retained. */
struct list_cell_t* cons (char *word, struct list_cell_t *lst)
{
  /* Allocate memory for the word. Do not forget the room for the final
     '\0' that is never counted by 'strlen'. */
  char *wmem = malloc ((strlen (word) + 1) * sizeof (char)) ;
  if (wmem == NULL) return (NULL) ;    /* No enough memory. */
  /* Allocate memory for the new cell. */
  struct list_cell_t *tmp = malloc (sizeof (struct list_cell_t)) ;
  if (tmp == NULL) {
    /* No enough memory. Free the memory allocated for the word above. */
    free (wmem) ;
    return (NULL) ;
  }
  strcpy (wmem, word) ;  /* Copy the characters of the word. */
  tmp->word = wmem ;     /* Store the address of the word. */
  tmp->next = lst ;      /* Link in head. */
  return (tmp) ;
}


/** Free all the memory occupied by the cells of the list 'lst'. It also
    free the memory allocated for each word. */
void free_list (struct list_cell_t *lst)
{
  if (lst != NULL) {
    /* Free the tail of the list. */
    free_list (lst->next) ;
    /* Free the memory occupied by the word if some is stored in the cell. */
    if (lst->word != NULL) free (lst->word) ;
    free (lst) ;
  }
}


/** Print the words of the linked-list 'lst', all on the same line. */
void print_list (struct list_cell_t *lst)
{
  while (lst != NULL) {
    printf (" %s", lst->word) ;
    lst = lst->next ;
  }
  printf ("\n") ;
}


int main (int argc, char *argv[])
{
  FILE *in_handle ;  /* Input file descriptor. */
  /* Address of the first cell of the list of words. */
  struct list_cell_t *list_head = NULL ;
  /* Temporary buffer to read a word from the file. We assume that read words
     will never be longer than 254 characters. */
  char buffer[255] ;

  /* Check the right number of command line argument(s). Only 1 expected. */
  if (argc != 2) {
    printf ("Error. Missing input file.\n") ;
    return (-1) ;
  }

  /* Attempt to open the file where to read the words. */
  in_handle = fopen (argv[1], "rb") ;
  if (in_handle == NULL) {
    printf ("Error. Unable to open the input file '%s'.\n", argv[1]) ;
    return (-1) ;
  }

  fscanf (in_handle, "%s", buffer) ;
  /* Loop until the end of fiel is reached. Attention, 'feof' returns 'false'
     only after a read was attempted! */
  while (! feof (in_handle)) {
    struct list_cell_t *new_head = cons (buffer, list_head) ;
    if (new_head == NULL) {
      /* Insertion failed. We must free the already allocated list. */
      free_list (list_head) ;
      /* Close the file. */
      fclose (in_handle) ;
      printf ("Error. Insertion failed.\n") ;
      return (-1) ;
    }
    /* Else, insertion succeeded. */
    list_head = new_head ;
    /* Read the next word if some exists. */
    fscanf (in_handle, "%s", buffer) ;
  }

  fclose (in_handle) ;
  print_list (list_head) ;
  free_list (list_head) ;
  return (0) ;
}
