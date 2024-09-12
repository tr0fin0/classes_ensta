#include "utils.h"
#include "display.h"

#ifdef _MPI_SUPPORT_
const int MASTER = 0;
enum tag {RESULT, DATA, TERMINATE};
extern int world_size, job_width, data_size;

void gui_display(int* results);
void gui_draw(int col, int* color);
#endif

inline int calc_pixel(ComplexNum& c)
{
    int repeats = 0;
    double lengthsq = 0.0;
    ComplexNum z = {0, 0};
    while (repeats < 100000 && lengthsq < 4.0) {
        double temp = z.real * z.real - z.imag * z.imag + c.real;
        z.imag = 2 * z.real * z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real * z.real + z.imag * z.imag;
        repeats++;
    }
    return repeats;
}