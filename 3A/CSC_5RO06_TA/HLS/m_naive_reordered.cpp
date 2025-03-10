#include "m_naive.h"

// Algoritmo 2: Multiplicación naive con bucles reordenados
void naiveReorderedMultiplication(hls::stream<axis> &input_stream, hls::stream<axis> &output_stream) {
	#pragma HLS INTERFACE ap_ctrl_none port=return                 // disable unwanted messages
	#pragma HLS INTERFACE axis register both port=output_stream    //
	#pragma HLS INTERFACE axis register both port=input_stream     //

	// matrix declaration
	int A[SIZE][SIZE];    // input
	int B[SIZE][SIZE];    // input
	int C[SIZE][SIZE];    // output

	// local variables
	int row, col, k;
	axis data_axis;

	loop_stream_read_row_A: for(row = 0; row < SIZE; row++){
		loop_stream_read_col_A: for(col = 0; col < SIZE; col++){
			A[row][col] = input_stream.read().data;
		}
	}
	// matrix B
	loop_stream_read_row_B: for(row = 0; row < SIZE; row++){
		loop_stream_read_col_B: for(col = 0; col < SIZE; col++){
			B[row][col] = input_stream.read().data;
		}
	}

	// benchmark function
	loop_mult_row: for (int i = 0; i < SIZE; ++i)
		loop_mult_k: for (int k = 0; k < SIZE; ++k)
			loop_mult_col: for (int j = 0; j < SIZE; ++j)
				C[i][j] += A[i][k] * B[k][j];

	//stream data write
	loop_stream_write_row: for(row = 0; row < SIZE; row++){
		loop_stream_write_col: for(col = 0; col < SIZE; col++){
			data_axis.data = C[row][col];

			if((row == SIZE-1) && (col == SIZE-1))
				data_axis.last = 1;
			else
				data_axis.last = 0;
			output_stream.write(data_axis);
		}
	}
}
