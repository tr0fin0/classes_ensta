void compute_mandelbrot_subset(int* result, int iter_limit, int x_resolution, int y_resolution, double x_begin, double y_begin) {
    
    double x_step = (x_end-x_begin) / (x_resolution-1);
    double y_step = (y_end-y_begin) / (y_resolution-1);
    
    int i, j;
    complex<double> c, z;

    for (i = 0; i < x_resolution * y_resolution; i++) {
        c = complex<double>(x_begin + (i % x_resolution) * x_step, y_begin + (i / x_resolution) * y_step );
        z = 0; j = 0;

        while (norm(z) <= 4 && j < iter_limit) { z = z*z + c; j++; }
            result[i-start] = j;
    }
}


void compute_mandelbrot_subset(int* result, int iter_limit, int start, int end) {
    int i, j;
    complex<double> c, z;
    # pragma omp parallel shared(result, iter_limit, start, end) private(i, j, c, z)
    # pragma omp for schedule(runtime)

    for (i = start; i < end; i++) {
        c = complex<double>(x_begin + (i % x_resolution) * x_step, y_begin + (i / x_resolution) * y_step );
        z = 0; j = 0;
        while (norm(z) <= 4 && j < iter_limit) { z = z*z + c; j++; }
            result[i-start] = j;
    }
}


// distribute tasks
    int start, end = 0;
    for (int i = 1; i < processors_amount; i++) {
        start = end; end += part_width;
        int message[2] = {start, end};
        MPI_Send(message, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    compute_mandelbrot_subset(current+end, iter_limit, end, result_size);
// join pieces together
    for (int i = 1; i < processors_amount; i++) {
        MPI_Recv(current, part_width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        current += part_width;
    }

1 int message[2];

2 MPI_Recv(message, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

3 int* partial_result = new int[part_width];

4 compute_mandelbrot_subset(partial_result, iter_limit, message[0], message[1]);

5 MPI_Send(partial_result, part_width, MPI_INT, 0, 0, MPI_COMM_WORLD);

6 delete[] partial_result;



1 compute_mandelbrot_subset(partial_result, iter_limit, start, start+part_width);

2 MPI_Gather(partial_result, part_width, MPI_INT, result, part_width, MPI_INT, 0, MPI_COMM_WORLD);




1 #include <ctime>

2 #include <cstdlib>

3

4 double now() {

5 struct timespec tp;

6 if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp) != 0) exit(1);

7 return tp.tv_sec + tp.tv_nsec / (double)1000000000;

8 }