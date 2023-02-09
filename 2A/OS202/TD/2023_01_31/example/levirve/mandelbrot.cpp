#include <mpi.h>
#include "mandelbrot_utils.h"
using namespace std;

int num_thread, width, height;
double dx, dy, real_min, imag_min, s;

int world_size, job_width, data_size, rank_num, *result, *results;

void start(int sz)
{
    ComplexNum c;
    for (int i = 0, x = rank_num * sz; i < sz && x < width; ++i, ++x) {
        c.real = x * dx + real_min;
        for (int j = 0; j < height; j++) {
            c.imag = j * dy + imag_min;
            result[j * sz + i] = calc_pixel(c);
        }
    }
}

void initial_MPI_env(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_num);

    job_width = width / world_size;
    if (width % world_size) job_width += 1;
    data_size = job_width * height;
    result = new int [data_size];
}

void collect_results()
{
    if (rank_num == 0) results = new int [world_size * data_size];
    MPI_Gather(result, data_size, MPI_INT, results, data_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    cout << fixed << rank_num << ": " << MPI_Wtime() - s << endl;
    MPI_Finalize();
}

int main(int argc, char** argv) {
    try {
        initial_env(argc, argv);
        initial_MPI_env(argc, argv);
        s = MPI_Wtime();
        start(job_width);
        collect_results();
        if (rank_num == 0 && gui) gui_display(results);
    } catch (char const* err) {
        cerr << err << endl;
    }
    delete [] results;
    delete [] result;
    return 0;
}
