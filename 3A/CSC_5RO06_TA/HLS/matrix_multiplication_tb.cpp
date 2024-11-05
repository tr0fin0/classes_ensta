#include "matrix_multiplication_tb.h"

#include <stdio.h>
#include <math.h>
#include <cstdlib>

int main(){
    // matrix declaration
    int A[SIZE][SIZE];    // input
    int B[SIZE][SIZE];    // input
    int C[SIZE][SIZE];  // output naive calculation

    // local variables
    int row, col;
    axis data_axis;

    // create A and B matrix
    loop_rand_row: for(row = 0; row < SIZE; row++){
        loop_rand_col: for(col = 0; col < SIZE; col++){
            A[row][col] = rand()%10;
            B[row][col] = rand()%10;
        }
    }

    //stream data write
    hls::stream<axis> input_stream, output_stream;

    loop_stream_write_A_row: for(row = 0; row < SIZE; row++){
        loop_stream_write_A_col: for(col = 0; col < SIZE; col++){
            data_axis.data = A[row][col];
            if((row == SIZE-1) && (col == SIZE-1))
                data_axis.last = 1;
            else
                data_axis.last = 0;
            input_stream.write(data_axis);
        }
    }
    loop_stream_write_B_row: for(row = 0; row < SIZE; row++){
        loop_stream_write_B_col: for(col = 0; col < SIZE; col++){
            data_axis.data = B[row][col];
            if((row == SIZE-1) && (col == SIZE-1))
                data_axis.last = 1;
            else
                data_axis.last = 0;
            input_stream.write(data_axis);
        }
    }

    // call benchmark function
	#ifdef solution_naive
	naiveMultiplication(input_stream, output_stream);
	#endif

	#ifdef solution_naive_reordered
	naiveReorderedMultiplication(input_stream, output_stream);
	#endif

	#ifdef solution_block
	blockMultiplication(input_stream, output_stream);
	#endif

	#ifdef solution_block_reordered
	blockReorderedMultiplication(input_stream, output_stream);
	#endif


    // stream data read
    loop_stream_read_row: for(row = 0; row < SIZE; row++){
        loop_stream_read_col: for(col = 0; col < SIZE; col++){
            C[row][col] = output_stream.read().data;
        }
    }
}
