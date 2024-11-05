#include <hls_stream.h>
#include <ap_int.h>

// configuration parameters for the simulation

#define SIZE 32
#define BLOCK_SIZE 2

// define axis data structure
struct axis{
    int data;
    ap_uint<1> last;
};

#define solution_block
// possibilities: solution_naive, solution_naive_reordered, solution_block, solution_block_reordered
