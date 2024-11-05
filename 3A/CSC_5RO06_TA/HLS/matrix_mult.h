#include <stdio.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// define matrix data type
typedef int matrix;

// define axis data structure
typedef ap_axis<32,4,5,5> axis; // 32-bit integer with side-channel.

#ifdef solution_0_0
void matrix_mult_0_0(hls::stream<axis> &input_stream, hls::stream<axis> &output_stream);
#endif

#ifdef solution_0_1
void matrix_mult_0_1(hls::stream<axis> &input_stream, hls::stream<axis> &output_stream);
#endif

#ifdef solution_0_2
void matrix_mult_0_2(hls::stream<axis> &input_stream, hls::stream<axis> &output_stream);
#endif

#ifdef solution_0_3
void matrix_mult_0_3(hls::stream<axis> &input_stream, hls::stream<axis> &output_stream);
#endif
